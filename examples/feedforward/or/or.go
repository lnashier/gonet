package or

import (
	"context"
	"fmt"
	"github.com/lnashier/gonet/feedforward"
	"github.com/lnashier/gonet/fns"
	"github.com/lnashier/gonet/help"
)

// Build creates and train an OR function
func Build(ctx context.Context) {
	nn := feedforward.New(
		feedforward.Shapes([]int{2, 4, 1}),
		feedforward.Activation(fns.Sigmoid),
		feedforward.ActivationDerivative(fns.SigmoidDerivative),
		feedforward.LearningRate(0.01),
	)

	fmt.Println(nn.String())

	inputs := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	targets := [][]float64{
		{0},
		{1},
		{1},
		{1},
	}

	help.Train(ctx, nn, 1000000, inputs, targets)

	for _, input := range inputs {
		output := nn.Predict(input)
		fmt.Println(input, "->", output)
	}
}
