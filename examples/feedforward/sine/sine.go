package sine

import (
	"context"
	"fmt"
	"github.com/lnashier/gonet/feedforward"
	"github.com/lnashier/gonet/fns"
	"github.com/lnashier/gonet/help"
	"math"
	"math/rand"
)

func getModel(name string) (*feedforward.Network, bool) {
	nn, _ := help.LoadFeedforward(
		name,
		feedforward.Activation(fns.Tanh),
		feedforward.ActivationDerivative(fns.TanhDerivative),
		feedforward.LearningRate(0.1),
	)
	if nn == nil {
		return feedforward.New(
			feedforward.Shapes([]int{1, 100, 1}),
			feedforward.Activation(fns.Tanh),
			feedforward.ActivationDerivative(fns.TanhDerivative),
			feedforward.LearningRate(0.1),
		), false
	}
	return nn, true
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
// TODO requires more work
func Build(ctx context.Context, args []string) {
	nn, loaded := getModel("bin/sine")

	fmt.Println("Loaded", loaded)
	fmt.Println(nn.String())

	// resuming training or not trained
	if (len(args) > 0 && args[0] == "1") || !loaded {
		inputs, targets := trainingData()
		help.Train(ctx, nn, 10000, inputs, targets)
		if err := help.Save("bin/sine", nn); err != nil {
			panic(err)
		}
	}

	for i := 0; i < 10; i++ {
		input := []float64{rand.Float64() * 2 * math.Pi}
		output := nn.Predict(input)
		fmt.Printf("Input: %.6f, Predicted Sine: %.6f, True Sine: %.6f\n", input[0], output[0], math.Sin(input[0]))
	}
}
