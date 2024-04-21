package mnist

import (
	"context"
	"fmt"
	"github.com/lnashier/gonet"
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
		feedforward.ActivationDerivative(fns.SigmoidDerivative),
		feedforward.LearningRate(1),
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
			feedforward.Shapes([]int{28 * 28, 128, 10}),
			feedforward.LearningRate(1),
			feedforward.Activation(fns.Sigmoid),
			feedforward.ActivationDerivative(fns.SigmoidDerivative),
		), false
	}
	return nn, true
}

func test(ctx context.Context, nn gonet.Network, inputs [][][]uint8, outputs []uint8) {
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

	fmt.Println(nn.String())

	// not retraining
	if !loaded {
		trainingInputs, trainingOutputs, err := trainingData(10, args[0], args[1])
		if err != nil {
			panic(err)
		}
		help.Train(ctx, nn, 10, trainingInputs, trainingOutputs)

		err = help.Save("bin/mnist.gob", nn)
		if err != nil {
			panic(err)
		}
	}

	unseenInputs, unseenOutputs, err := readData(args[2], args[3])
	if err != nil {
		panic(err)
	}

	test(ctx, nn, unseenInputs, unseenOutputs)

	if len(args) > 4 && args[4] == "1" {
		oldShape := loadOldShape("bin/old/mnist_1.gob")

		//fmt.Println(oldShape)

		net := feedforward.New(
			feedforward.Shapes([]int{oldShape.InputSize, oldShape.HiddenSize, oldShape.OutputSize}),
			feedforward.Activation(fns.Sigmoid),
			feedforward.ActivationDerivative(fns.SigmoidDerivative),
		)

		fmt.Println(net.String())

		feedforward.ConvertOldShape(oldShape, net)

		fmt.Println(net.String())

		test(ctx, net, unseenInputs, unseenOutputs)
	}
}

func loadOldShape(name string) *feedforward.OldShape {
	file, err := os.Open(name)
	if err != nil {
		return nil
	}
	defer file.Close()

	oldShape, err := feedforward.LoadOldShape(file)
	if err != nil {
		panic(err)
	}

	return oldShape
}
