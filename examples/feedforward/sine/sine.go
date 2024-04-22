package sine

import (
	"context"
	"fmt"
	"github.com/lnashier/gonet/feedforward"
	"github.com/lnashier/gonet/fns"
	"github.com/lnashier/gonet/help"
	"math"
	"math/rand"
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
		feedforward.Activation(fns.ReLU),
		feedforward.ActivationDerivative(fns.ReLUDerivative),
		feedforward.LearningRate(0.1),
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
			feedforward.Shapes([]int{1, 100, 1}),
			feedforward.Activation(fns.ReLU),
			feedforward.ActivationDerivative(fns.ReLUDerivative),
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
	nn, loaded := getModel("bin/sine.gob")

	fmt.Println("loaded", loaded)
	fmt.Println(nn.String())

	// resuming training or not trained
	if (len(args) > 0 && args[0] == "1") || !loaded {
		inputs, targets := trainingData()
		help.Train(ctx, nn, 10000, inputs, targets)
		if err := help.Save("bin/sine.gob", nn); err != nil {
			panic(err)
		}
	}

	for i := 0; i < 10; i++ {
		input := []float64{rand.Float64() * 2 * math.Pi}
		output := nn.Predict(input)
		fmt.Printf("Input: %.6f, Predicted Sine: %.6f, True Sine: %.6f\n", input[0], output[0], math.Sin(input[0]))
	}
}
