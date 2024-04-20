package xor3

import (
	"context"
	"fmt"
	"github.com/lnashier/gonet"
	"github.com/lnashier/gonet/fns"
	"github.com/lnashier/gonet/help"
)

// Build trains a 3 variable XOR function
func Build(ctx context.Context) {
	// We want to model 3 variables XOR function
	inputSize := 3
	// Output is a single value 0 or 1
	outputSize := 1
	// Don't worry about this too much at this moment
	// It is simply saying how many nodes in first hidden layer
	// since only single layer FFN is supported
	hiddenSize := 4

	nn := gonet.Feedforward(
		gonet.InputSize(inputSize),   // defined above
		gonet.HiddenSize(hiddenSize), // defined above
		gonet.OutputSize(outputSize), // defined above
		// Sigmoid function squashes the output to the range [0, 1], that makes it suitable for binary classification.
		// tanh does [-1, 1]
		gonet.Activation(fns.Sigmoid),                     // this is chosen based on prior knowledge of training data.
		gonet.ActivationDerivative(fns.SigmoidDerivative), // backpropagation
		gonet.LearningRate(0.1),                           // Learning rate, choose wisely
	)

	fmt.Println(nn.String())

	trainingInputs := [][]float64{
		{0, 0, 0},
		{0, 0, 1},
		{0, 1, 0},
		{0, 1, 1},
		{1, 0, 0},
		{1, 0, 1},
		{1, 1, 0},
		{1, 1, 1},
	}
	targetOutputs := [][]float64{
		{0},
		{1},
		{1},
		{1},
		{1},
		{1},
		{1},
		{0},
	}

	help.Train(ctx, nn, 10000, trainingInputs, targetOutputs)

	for _, input := range trainingInputs {
		output := nn.Predict(input)
		fmt.Println(input, "->", output)
	}
}
