package sine

import (
	"context"
	"fmt"
	"github.com/lnashier/gonet"
	"github.com/lnashier/gonet/fns"
	"github.com/lnashier/gonet/help"
	"math"
	"math/rand"
)

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

// Build creates and train Sine function
func Build(ctx context.Context) {
	trainingInputs, targetOutputs := trainingData()

	nn := gonet.Feedforward(
		gonet.InputSize(1),
		gonet.HiddenSize(32),
		gonet.OutputSize(1),
		gonet.Activation(fns.ReLU),
		gonet.ActivationDerivative(fns.ReLUDerivative),
		gonet.LearningRate(0.1),
	)

	fmt.Println(nn.String())

	help.Train(ctx, nn, 100000, trainingInputs, targetOutputs)

	for i := 0; i < 10; i++ {
		input := []float64{rand.Float64() * 2 * math.Pi}
		output := nn.Predict(input)
		fmt.Printf("Input: %.6f, Predicted Sine: %.6f, True Sine: %.6f\n", input[0], output[0], math.Sin(input[0]))
	}
}
