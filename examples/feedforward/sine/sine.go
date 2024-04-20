package sine

import (
	"context"
	"fmt"
	"github.com/lnashier/gonet"
	"github.com/lnashier/gonet/fns"
	"github.com/lnashier/gonet/help"
	"math"
	"math/rand"
	"os"
)

func getModel(name string) (*gonet.FeedforwardNetwork, bool) {
	model, _ := os.ReadFile(name)
	return gonet.Feedforward(
		gonet.LoadFrom(model),
		gonet.InputSize(1),
		gonet.HiddenSize(32),
		gonet.OutputSize(1),
		gonet.Activation(fns.ReLU),
		gonet.ActivationDerivative(fns.ReLUDerivative),
		gonet.LearningRate(0.1),
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

func trainingData() ([][]float64, [][]float64) {
	samples := 10000
	trainingInputs := make([][]float64, samples)
	targetOutputs := make([][]float64, samples)

	for i := 0; i < samples; i++ {
		// Random angles in the range [0, 2*pi]
		trainingInputs[i] = []float64{rand.Float64() * 2 * math.Pi}
		targetOutputs[i] = []float64{math.Sin(trainingInputs[i][0])}
	}

	return trainingInputs, targetOutputs
}

// Build creates and trains Sine function
func Build(ctx context.Context) {
	nn, loaded := getModel("bin/sine.gob")

	fmt.Println(nn.String())

	if !loaded {
		trainingInputs, targetOutputs := trainingData()
		help.Train(ctx, nn, 100000, trainingInputs, targetOutputs)
		if err := saveModel("bin/sine.gob", nn); err != nil {
			panic(err)
		}
	}

	for i := 0; i < 10; i++ {
		input := []float64{rand.Float64() * 2 * math.Pi}
		output := nn.Predict(input)
		fmt.Printf("Input: %.6f, Predicted Sine: %.6f, True Sine: %.6f\n", input[0], output[0], math.Sin(input[0]))
	}
}
