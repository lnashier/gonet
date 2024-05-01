package xorclassification

import (
	"context"
	"fmt"
	"github.com/lnashier/gonet"
	"github.com/lnashier/gonet/feedforward"
	"github.com/lnashier/gonet/fns"
	"github.com/lnashier/gonet/help"
	"time"
)

func getModel(name string) (*feedforward.Network, bool) {
	nn, _ := help.LoadFeedforward(
		name,
		feedforward.Activation(fns.Sigmoid),
		feedforward.ActivationDerivative(fns.SigmoidDerivative),
		feedforward.LearningRate(0.25),
	)
	if nn == nil {
		return feedforward.New(
			feedforward.Shapes([]int{2, 4, 2}),
			feedforward.Activation(fns.Sigmoid),
			feedforward.ActivationDerivative(fns.SigmoidDerivative),
			feedforward.LearningRate(0.25),
		), false
	}
	return nn, true
}

func hotEncode(outputSize int, labels []int) [][]float64 {
	var outputs [][]float64
	for i := range labels {
		output := make([]float64, outputSize)
		output[labels[i]] = 1.0 // One-hot encode the label
		outputs = append(outputs, output)
	}
	return outputs
}

func test(ctx context.Context, nn gonet.Network, inputs [][]float64, targets []int) {
	var correctPredictions int
	var predictions int

	for i, input := range inputs {
		select {
		case <-ctx.Done():
			return
		default:
		}

		prediction := nn.Predict(input)
		fmt.Println(input, "->", prediction)
		predictions++
		output := fns.Argmax(prediction)
		if output == targets[i] {
			correctPredictions++
		}
	}

	fmt.Printf(
		"Total Predictions: %d, Correct Predictions: %d, Accuracy: %.2f%%\n",
		predictions,
		correctPredictions,
		(float64(correctPredictions)/float64(predictions))*100,
	)
}

// Build trains the network to predict XOR output as two classes 0 and 1.
// In this modified version, the output size is set to 2 to accommodate the one-hot encoded labels for the XOR.
// Each output node corresponds to a class, and the output values will represent the probabilities of belonging to each class.
func Build(ctx context.Context, args []string) {
	name := "bin/xorclassification"

	nn, loaded := getModel(name)

	fmt.Println("Loaded", loaded)
	fmt.Println(nn.String())

	if !loaded {
		if err := help.Save(fmt.Sprintf("%s-random-%s", name, time.Now().Format("060402150405")), nn); err != nil {
			panic(err)
		}
	}

	var inputs = [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}

	var labels = []int{
		0, // Class 0: [1, 0]
		1, // Class 1: [0, 1]
		1, // Class 1: [0, 1]
		0, // Class 0: [1, 0]
	}

	targets := hotEncode(2, labels)

	// resuming training or not trained
	if (len(args) > 0 && args[0] == "1") || !loaded {
		fmt.Println("Testing on training-data before (re)training")
		test(ctx, nn, inputs, labels)

		help.Train(ctx, nn, 10000, inputs, targets, help.EchoStatsEvery(5*time.Second), help.LossFunc(fns.LogLoss))

		if err := help.Save(name, nn); err != nil {
			panic(err)
		}
	}

	fmt.Println("Testing on training-data after (re)training")
	test(ctx, nn, inputs, labels)
}
