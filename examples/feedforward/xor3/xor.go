package xor3

import (
	"context"
	"fmt"
	"github.com/lnashier/gonet/feedforward"
	"github.com/lnashier/gonet/fns"
	"github.com/lnashier/gonet/help"
)

// Build trains a 3 variable XOR function
func Build(ctx context.Context) {
	// We want to model 3 variables XOR function
	inputSize := 3
	// Output is a single value 0 or 1
	outputSize := 1
	// Nodes in hidden layer
	hiddenSize := 4

	nn := feedforward.New(
		// defined above
		feedforward.Shapes([]int{inputSize, hiddenSize, outputSize}),

		// This is chosen based on prior knowledge of training data.
		// Sigmoid function squashes the output to the range [0, 1],
		// that makes it suitable for binary classification.
		feedforward.Activation(fns.Sigmoid),

		// backpropagation
		feedforward.ActivationDerivative(fns.SigmoidDerivative),

		// Learning rate, choose wisely
		feedforward.LearningRate(0.1),
	)

	fmt.Println(nn.String())

	inputs := [][]float64{
		{0, 0, 0},
		{0, 0, 1},
		{0, 1, 0},
		{0, 1, 1},
		{1, 0, 0},
		{1, 0, 1},
		{1, 1, 0},
		{1, 1, 1},
	}
	targets := [][]float64{
		{0},
		{1},
		{1},
		{1},
		{1},
		{1},
		{1},
		{0},
	}

	help.Train(ctx, nn, 100000, inputs, targets)

	for _, input := range inputs {
		output := nn.Predict(input)
		fmt.Println(input, "->", output)
	}
}
