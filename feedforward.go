package gonet

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"github.com/lnashier/gonet/fns"
	"io"
	"time"
)

// FeedforwardNetwork is an implementation of a Multilayer Perceptron (MLP).
type FeedforwardNetwork struct {
	ffn   *ffn
	lr    float64
	af    func(float64) float64
	fd    func(float64) float64
	stats *TrainingStats
}

type ffn struct {
	InputSize     int
	InputWeights  [][]float64
	InputBiases   []float64
	HiddenSize    int
	HiddenWeights [][]float64
	HiddenBiases  []float64
	OutputSize    int
}

func LoadFeedforward(src io.Reader) (*FeedforwardNetwork, error) {
	var fn ffn
	if err := gob.NewDecoder(src).Decode(&fn); err != nil {
		return nil, err
	}
	return &FeedforwardNetwork{ffn: &fn}, nil
}

// Feedforward initializes the weights and biases for the FeedforwardNetwork.
func Feedforward(opt ...NetworkOpt) *FeedforwardNetwork {
	opts := defaultNetworkOpts
	opts.apply(opt)

	var nn *FeedforwardNetwork
	if opts.loadFrom != nil {
		var err error
		nn, err = LoadFeedforward(bytes.NewBuffer(opts.loadFrom))
		if err != nil {
			panic(err)
		}
	} else {
		nn = &FeedforwardNetwork{
			ffn: &ffn{
				InputSize:     opts.inputSize,
				HiddenSize:    opts.hiddenSize,
				OutputSize:    opts.outputSize,
				InputWeights:  fns.RandomMatrix(opts.inputSize, opts.hiddenSize),
				InputBiases:   fns.RandomVector(opts.hiddenSize),
				HiddenWeights: fns.RandomMatrix(opts.hiddenSize, opts.outputSize),
				HiddenBiases:  fns.RandomVector(opts.outputSize),
			},
		}
	}

	nn.lr = opts.learningRate
	nn.af = opts.activation
	nn.fd = opts.activationDerivative

	return nn
}

func (nn *FeedforwardNetwork) Save(w io.Writer) error {
	return gob.NewEncoder(w).Encode(nn.ffn)
}

func (nn *FeedforwardNetwork) forward(input []float64) ([]float64, []float64) {
	hiddenActivations := make([]float64, nn.ffn.HiddenSize)

	for i := range hiddenActivations {
		sum := nn.ffn.InputBiases[i]

		for j := range input {
			sum += input[j] * nn.ffn.InputWeights[j][i]
		}

		hiddenActivations[i] = nn.af(sum)
	}

	output := make([]float64, nn.ffn.OutputSize)
	for i := range output {
		sum := nn.ffn.HiddenBiases[i]

		for j := range hiddenActivations {
			sum += hiddenActivations[j] * nn.ffn.HiddenWeights[j][i]
		}

		output[i] = nn.af(sum)
	}

	return hiddenActivations, output
}

func (nn *FeedforwardNetwork) backward(input, targetOutput, hiddenActivations, output []float64) {
	outputDelta := make([]float64, nn.ffn.OutputSize)
	for i := range outputDelta {
		outputDelta[i] = (targetOutput[i] - output[i]) * nn.fd(output[i])
	}

	hiddenDelta := make([]float64, nn.ffn.HiddenSize)
	for i := range hiddenDelta {
		hiddenErr := 0.0
		for j := range nn.ffn.HiddenWeights[i] {
			hiddenErr += outputDelta[j] * nn.ffn.HiddenWeights[i][j]
		}
		hiddenDelta[i] = hiddenErr * nn.fd(hiddenActivations[i])
	}

	for i := range nn.ffn.HiddenWeights {
		for j := range nn.ffn.HiddenWeights[i] {
			nn.ffn.HiddenWeights[i][j] += nn.lr * hiddenActivations[i] * outputDelta[j]
		}
	}

	for i := range nn.ffn.HiddenBiases {
		nn.ffn.HiddenBiases[i] += nn.lr * outputDelta[i]
	}

	for i := range nn.ffn.InputWeights {
		for j := range nn.ffn.InputWeights[i] {
			nn.ffn.InputWeights[i][j] += nn.lr * input[i] * hiddenDelta[j]
		}
	}

	for i := range nn.ffn.InputBiases {
		nn.ffn.InputBiases[i] += nn.lr * hiddenDelta[i]
	}
}

func (nn *FeedforwardNetwork) Predict(input []float64) []float64 {
	_, output := nn.forward(input)
	return output
}

func (nn *FeedforwardNetwork) Train(epochs int, inputs, outputs [][]float64, callback func(int) bool) {
	nn.stats = &TrainingStats{
		Start:  time.Now(),
		Epochs: make(map[int]*EpochStats),
	}
	defer func() {
		nn.stats.End = time.Now()
	}()

	for epoch := range epochs {
		nn.stats.Epochs[epoch] = &EpochStats{
			ID:       epoch,
			Start:    time.Now(),
			Forward:  &StageStats{},
			Backward: &StageStats{},
		}

		for i, input := range inputs {
			nn.stats.Epochs[epoch].Inputs++

			forwardStart := time.Now()
			hiddenActivations, output := nn.forward(input)
			nn.stats.Epochs[epoch].Forward.Duration += time.Since(forwardStart)
			nn.stats.Epochs[epoch].Forward.Count++

			backwardStart := time.Now()
			nn.backward(input, outputs[i], hiddenActivations, output)
			nn.stats.Epochs[epoch].Backward.Duration += time.Since(backwardStart)
			nn.stats.Epochs[epoch].Backward.Count++
		}

		nn.stats.Epochs[epoch].End = time.Now()

		if !callback(epoch) {
			break
		}
	}
}

func (nn *FeedforwardNetwork) String() string {
	return fmt.Sprintf(
		"Input Size: %d\n"+
			"Hidden Size: %d\n"+
			"Output Size: %d\n"+
			"Learning Rate: %.4f\n",
		nn.ffn.InputSize,
		nn.ffn.HiddenSize,
		nn.ffn.OutputSize,
		nn.lr,
	)
}

func (nn *FeedforwardNetwork) TrainingDuration() time.Duration {
	if nn.stats == nil || nn.stats.Start.IsZero() {
		return -1
	}
	if nn.stats.End.IsZero() {
		return time.Since(nn.stats.Start)
	}
	return nn.stats.End.Sub(nn.stats.Start)
}

func (nn *FeedforwardNetwork) EpochDuration(epoch int) time.Duration {
	if nn.stats == nil {
		return -1
	}
	epochStats, ok := nn.stats.Epochs[epoch]
	if ok {
		if epochStats.Start.IsZero() {
			return -1
		}
		if epochStats.End.IsZero() {
			return time.Since(epochStats.Start)
		}
		return epochStats.End.Sub(epochStats.Start)
	}
	return -1
}

func (nn *FeedforwardNetwork) EpochStats(epoch int) EpochStats {
	if nn.stats == nil {
		return EpochStats{}
	}
	epochStats, ok := nn.stats.Epochs[epoch]
	if ok {
		return *epochStats
	}
	return EpochStats{}
}

type TrainingStats struct {
	Start  time.Time
	End    time.Time
	Epochs map[int]*EpochStats
}

type EpochStats struct {
	ID       int
	Start    time.Time
	End      time.Time
	Inputs   int
	Forward  *StageStats
	Backward *StageStats
}

type StageStats struct {
	Duration time.Duration
	Count    int
}
