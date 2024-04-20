package mnist

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
		gonet.InputSize(28*28),
		gonet.HiddenSize(128),
		gonet.OutputSize(10),
		gonet.LearningRate(1),
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

func test(ctx context.Context, nn *gonet.FeedforwardNetwork, inputs [][][]uint8, outputs []uint8) {
	var correctPredictions int
	var predictions int

	defer func() {
		accuracy := float64(correctPredictions) / float64(predictions)
		fmt.Printf("Total Predictions: %d, Correct Predictions: %d, Accuracy: %.2f%%\n", predictions, correctPredictions, accuracy*100)
	}()

	for i, input := range inputs {
		select {
		case <-ctx.Done():
			return
		default:
		}
		prediction := nn.Predict(flattenImage(input))
		predictions++
		output := fns.Argmax(prediction)
		if output == int(outputs[i]) {
			err := saveImage(input, fmt.Sprintf("bin/correct/image-p%d-r%d.png", output, outputs[i]))
			if err != nil {
				panic(err)
			}
			correctPredictions++
		} else {
			err := saveImage(input, fmt.Sprintf("bin/wrong/image-p%d-r%d.png", output, outputs[i]))
			if err != nil {
				panic(err)
			}
		}
	}
}

func Build(ctx context.Context, args []string) {
	nn, loaded := getModel("bin/mnist.gob")
	if !loaded {
		trainingInputs, trainingOutputs, err := trainingData(10, args[0], args[1])
		if err != nil {
			panic(err)
		}
		help.Train(ctx, nn, 10, trainingInputs, trainingOutputs)

		err = saveModel("bin/mnist.gob", nn)
		if err != nil {
			panic(err)
		}
	}

	unseenInputs, unseenOutputs, err := readData(args[2], args[3])
	if err != nil {
		panic(err)
	}
	test(ctx, nn, unseenInputs, unseenOutputs)
}
