package or

import (
	"context"
	"fmt"
	"github.com/lnashier/gonet"
	"github.com/lnashier/gonet/fns"
	"github.com/lnashier/gonet/help"
)

// Build creates and train an OR function
func Build(ctx context.Context) {
	nn := gonet.Feedforward(
		gonet.InputSize(2),
		gonet.HiddenSize(4),
		gonet.OutputSize(1),
		gonet.Activation(fns.Sigmoid),
		gonet.ActivationDerivative(fns.SigmoidDerivative),
		gonet.LearningRate(1),
	)

	fmt.Println(nn.String())

	trainingInputs := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	targetOutputs := [][]float64{
		{0},
		{1},
		{1},
		{1},
	}

	help.Train(ctx, nn, 10000, trainingInputs, targetOutputs)

	for _, input := range trainingInputs {
		output := nn.Predict(input)
		fmt.Println(input, "->", output)
	}
}
