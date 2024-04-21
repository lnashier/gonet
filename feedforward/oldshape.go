package feedforward

import (
	"encoding/gob"
	"io"
)

type OldShape struct {
	InputSize     int
	InputWeights  [][]float64
	InputBiases   []float64
	HiddenSize    int
	HiddenWeights [][]float64
	HiddenBiases  []float64
	OutputSize    int
}

func LoadOldShape(src io.Reader) (*OldShape, error) {
	var shape OldShape
	if err := gob.NewDecoder(src).Decode(&shape); err != nil {
		return nil, err
	}
	return &shape, nil
}

func ConvertOldShape(shape *OldShape, net *Network) {
	net.Layers[0] = &Layer{
		Nodes: make([]*Node, shape.HiddenSize),
	}

	for i, b := range shape.InputBiases {
		node := &Node{
			Weights: make([]float64, shape.InputSize),
			Bias:    b,
		}
		for j, weights := range shape.InputWeights {
			node.Weights[j] = weights[i]
		}
		net.Layers[0].Nodes[i] = node
	}

	net.Layers[1] = &Layer{
		Nodes: make([]*Node, shape.OutputSize),
	}

	for i, b := range shape.HiddenBiases {
		node := &Node{
			Weights: make([]float64, shape.HiddenSize),
			Bias:    b,
		}
		for j, weights := range shape.HiddenWeights {
			node.Weights[j] = weights[i]
		}
		net.Layers[1].Nodes[i] = node
	}
}
