package xor

import (
	"context"
	"fmt"
	"github.com/lnashier/gonet/feedforward"
	"github.com/lnashier/gonet/fns"
	"github.com/lnashier/gonet/help"
	"os"
)

func loadModel(name string) *feedforward.Network {
	model, err := os.Open(name)
	if err != nil {
		return nil
	}
	defer model.Close()

	nn, err := feedforward.Load(
		model,
		feedforward.Activation(fns.Sigmoid),
		// network may be retrained (resume training)
		feedforward.ActivationDerivative(fns.SigmoidDerivative),
		feedforward.LearningRate(0.01),
	)
	if err != nil {
		panic(err)
	}
	return nn
}

func getModel(name string) (*feedforward.Network, bool) {
	nn := loadModel(name)
	if nn == nil {
		return feedforward.New(
			feedforward.Shapes([]int{2, 4, 1}),
			feedforward.LearningRate(0.5),
			feedforward.Activation(fns.Sigmoid),
			feedforward.ActivationDerivative(fns.SigmoidDerivative),
		), false
	}
	return nn, true
}

// Build trains a XOR function
func Build(ctx context.Context, args []string) {
	nn, loaded := getModel("bin/xor.gob")

	fmt.Println("loaded", loaded)
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
		if err := help.Save("bin/xor.gob", nn); err != nil {
			panic(err)
		}
	}

	for _, input := range inputs {
		output := nn.Predict(input)
		fmt.Println(input, "->", output)
	}
}
