package feedforward

import (
	"encoding/gob"
	"fmt"
	"github.com/lnashier/gonet/fns"
	"github.com/lnashier/gonet/stats"
	"io"
	"math/rand"
	"time"
)

type Node struct {
	Weights []float64
	Bias    float64
}

type Layer struct {
	Nodes []*Node
}

type Network struct {
	shapes []int
	layers []*Layer
	stats  *stats.Training
	af     func(float64) float64
	fd     func(float64) float64
	lr     float64
}

func Load(src io.Reader, opt ...NetworkOpt) (*Network, error) {
	var layers []*Layer
	if err := gob.NewDecoder(src).Decode(&layers); err != nil {
		return nil, err
	}

	opts := defaultNetworkOpts
	opts.apply(opt)

	nn := &Network{layers: layers}
	nn.shapes = make([]int, len(nn.layers)+1)
	nn.shapes[0] = len(nn.layers[0].Nodes[0].Weights)
	for i, layer := range nn.layers {
		nn.shapes[i+1] = len(layer.Nodes)
	}
	// prediction
	nn.af = opts.activation
	// network may be retrained (resume training)
	nn.fd = opts.activationDerivative
	nn.lr = opts.learningRate

	return nn, nil
}

func New(opt ...NetworkOpt) *Network {
	opts := defaultNetworkOpts
	opts.apply(opt)

	nn := &Network{shapes: opts.shapes}

	// input, hidden(s), output
	nn.layers = make([]*Layer, len(nn.shapes)-1)

	for i := 1; i < len(nn.shapes); i++ {
		layer := &Layer{
			Nodes: make([]*Node, nn.shapes[i]),
		}

		for j := range layer.Nodes {
			layer.Nodes[j] = &Node{
				Weights: fns.RandomVector(nn.shapes[i-1]),
				Bias:    rand.Float64() - 0.5,
			}
		}

		nn.layers[i-1] = layer
	}

	nn.af = opts.activation
	nn.fd = opts.activationDerivative
	nn.lr = opts.learningRate

	return nn
}

func (nn *Network) Train(epochs int, inputs, targets [][]float64, callback func(int) bool) {
	nn.stats = &stats.Training{
		Start: time.Now(),
	}
	defer func() {
		nn.stats.End = time.Now()
	}()

	for epoch := range epochs {
		epochStat := &stats.Epoch{
			ID:    epoch,
			Start: time.Now(),
		}
		nn.stats.Epochs.Store(epoch, epochStat)
		for i, input := range inputs {
			epochStat.Inputs++
			nn.backward(input, targets[i])
		}
		epochStat.End = time.Now()
		if !callback(epoch) {
			break
		}
	}
}

func (nn *Network) String() string {
	return fmt.Sprintf("Shapes: %v\nHidden Layers: %d\n", nn.shapes, len(nn.layers)-1)
}

func (nn *Network) TrainingDuration() time.Duration {
	if nn.stats == nil || nn.stats.Start.IsZero() {
		return -1
	}
	if nn.stats.End.IsZero() {
		return time.Since(nn.stats.Start)
	}
	return nn.stats.End.Sub(nn.stats.Start)
}

func (nn *Network) EpochStats(epoch int) stats.Epoch {
	if nn.stats == nil {
		return stats.Epoch{}
	}
	epochStats, ok := nn.stats.Epochs.Load(epoch)
	if ok {
		return *epochStats.(*stats.Epoch)
	}
	return stats.Epoch{}
}

func (nn *Network) Predict(input []float64) []float64 {
	activation := input
	for l := range len(nn.layers) {
		activation = nn.forward(activation, l)
	}
	return activation
}

func (nn *Network) Save(w io.Writer) error {
	layers := nn.layers
	return gob.NewEncoder(w).Encode(layers)
}

func (nn *Network) forward(prevActivation []float64, atLayer int) []float64 {
	layer := nn.layers[atLayer]
	activation := make([]float64, len(layer.Nodes))
	for i, node := range layer.Nodes {
		sum := node.Bias
		for j := range prevActivation {
			sum += prevActivation[j] * node.Weights[j]
		}
		activation[i] = nn.af(sum)
	}
	return activation
}

func (nn *Network) backward(input []float64, target []float64) {
	activations := make([][]float64, len(nn.layers))
	activation := input
	for l := range len(nn.layers) {
		activation = nn.forward(activation, l)
		activations[l] = activation
	}

	deltas := make([][]float64, len(nn.layers))

	// deltas
	for l := len(nn.layers) - 1; l >= 0; l-- {
		currLayer := nn.layers[l]
		currActivation := activations[l]
		var nextLayer *Layer
		if l+1 < len(nn.layers) {
			nextLayer = nn.layers[l+1]
		}

		deltas[l] = make([]float64, len(currLayer.Nodes))

		for i := range deltas[l] {
			err := 0.0
			if nextLayer != nil {
				// current is a hidden layer
				for j, node := range nextLayer.Nodes {
					err += deltas[l+1][j] * node.Weights[i]
				}
			} else {
				// current is the output layer
				err = target[i] - currActivation[i]
			}
			deltas[l][i] = err * nn.fd(currActivation[i])
		}
	}

	// update
	for l := len(nn.layers) - 1; l >= 0; l-- {
		var prevLayer *Layer
		var prevActivation []float64
		if l-1 >= 0 {
			prevLayer = nn.layers[l-1]
			prevActivation = activations[l-1]
		}
		currLayer := nn.layers[l]

		for i, currNode := range currLayer.Nodes {
			delta := deltas[l][i]
			currNode.Bias = nn.lr * delta

			if prevLayer != nil {
				// current layer represents either the output layer or a hidden layer beyond the first
				for j, _ := range prevLayer.Nodes {
					currNode.Weights[j] += nn.lr * prevActivation[j] * delta
				}
			} else {
				// current layer is the first hidden layer
				for j := range len(input) {
					currNode.Weights[j] += nn.lr * input[j] * delta
				}
			}
		}
	}
}
