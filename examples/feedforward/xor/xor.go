package xor

import (
	"context"
	"fmt"
	"github.com/lnashier/gonet"
	"github.com/lnashier/gonet/fns"
	"github.com/lnashier/gonet/help"
	"os"
)

func getModel(name string) (*gonet.FeedforwardNetwork, bool) {
	model, _ := os.ReadFile(name)
	return gonet.Feedforward(
		gonet.LoadFrom(model),
		gonet.InputSize(2),
		gonet.HiddenSize(4),
		gonet.OutputSize(1),
		gonet.LearningRate(0.1),
		gonet.Activation(fns.Sigmoid),
		gonet.ActivationDerivative(fns.SigmoidDerivative),
	), len(model) > 0
}

func saveModel(name string, nn *gonet.FeedforwardNetwork) error {
	file, err := os.Create(name)
	if err != nil {
		panic(err)
	}
	defer file.Close()
	return nn.Save(file)
}

// Build trains a XOR function
func Build(ctx context.Context) {
	var trainingData = [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}

	var targetOutputs = [][]float64{
		{0},
		{1},
		{1},
		{0},
	}

	nn, loaded := getModel("bin/xor.gob")
	if !loaded {
		help.Train(ctx, nn, 10000000, trainingData, targetOutputs)

		err := saveModel("bin/xor.gob", nn)
		if err != nil {
			panic(err)
		}
	}

	for _, input := range trainingData {
		output := nn.Predict(input)
		fmt.Println(input, "->", output)
	}
}
