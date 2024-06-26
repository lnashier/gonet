package xor

import (
	"context"
	"fmt"
	"github.com/lnashier/gonet/feedforward"
	"github.com/lnashier/gonet/fns"
	"github.com/lnashier/gonet/help"
)

func getModel(name string) (*feedforward.Network, bool) {
	nn, _ := help.LoadFeedforward(
		name,
		feedforward.Activation(fns.Sigmoid),
		feedforward.ActivationDerivative(fns.SigmoidDerivative),
		feedforward.LearningRate(0.01),
	)
	if nn == nil {
		return feedforward.New(
			feedforward.Shapes([]int{2, 4, 1}),
			feedforward.Activation(fns.Sigmoid),
			feedforward.ActivationDerivative(fns.SigmoidDerivative),
			feedforward.LearningRate(0.01),
		), false
	}
	return nn, true
}

// Build trains a XOR function
// In this scenario, we're predicting the output value of the XOR function, which can be either 0 or 1,
// but our neural network will output continuous values between 0 and 1.
func Build(ctx context.Context, args []string) {
	nn, loaded := getModel("bin/xor")

	fmt.Println("Loaded", loaded)
	fmt.Println(nn.String())

	var inputs = [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}

	var targets = [][]float64{
		{0},
		{1},
		{1},
		{0},
	}

	// resuming training or not trained
	if (len(args) > 0 && args[0] == "1") || !loaded {
		help.Train(ctx, nn, 100000, inputs, targets)
		if err := help.Save("bin/xor", nn); err != nil {
			panic(err)
		}
	}

	for _, input := range inputs {
		output := nn.Predict(input)
		fmt.Println(input, "->", output)
	}
}
