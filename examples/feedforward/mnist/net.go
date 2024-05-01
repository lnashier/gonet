package mnist

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
		feedforward.LearningRate(0.1),
	)
	if nn == nil {
		return feedforward.New(
			feedforward.Shapes([]int{28 * 28, 128, 10}),
			feedforward.Activation(fns.Sigmoid),
			feedforward.ActivationDerivative(fns.SigmoidDerivative),
			feedforward.LearningRate(0.1),
		), false
	}
	return nn, true
}

func test(ctx context.Context, nn gonet.Network, inputs [][][]uint8, targets []uint8) {
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
		if output == int(targets[i]) {
			correctPredictions++
		} else {
			// uncomment to save wrong predictions
			/*
				err := saveImage(input, fmt.Sprintf("bin/wrong/image_%d-p%d-r%d.png", i, output, targets[i]))
				if err != nil {
					panic(err)
				}
			*/
		}
	}
}

func Build(ctx context.Context, args []string) {
	name := "bin/mnist"

	nn, loaded := getModel(name)

	fmt.Println("Loaded", loaded)
	fmt.Println(nn.String())

	if !loaded {
		if err := help.Save(fmt.Sprintf("%s-random-%s", name, time.Now().Format("060402150405")), nn); err != nil {
			panic(err)
		}
	}

	// resuming training or not trained
	if (len(args) > 4 && args[4] == "1") || !loaded {
		images, labels, err := readData(args[0], args[1])
		if err != nil {
			panic(err)
		}

		fmt.Printf("Normalizing %d images and %d labels\n", len(images), len(labels))

		inputs, targets, err := normalizeData(10, images, labels)
		if err != nil {
			panic(err)
		}

		fmt.Println("Testing on training-data before (re)training")
		test(ctx, nn, images, labels)

		help.Train(ctx, nn, 10, inputs, targets, help.LossFunc(fns.LogLoss))

		err = help.Save(name, nn)
		if err != nil {
			panic(err)
		}

		fmt.Println("Testing on training-data after (re)training")
		test(ctx, nn, images, labels)
	}

	unseenInputs, unseenTargets, err := readData(args[2], args[3])
	if err != nil {
		panic(err)
	}

	test(ctx, nn, unseenInputs, unseenTargets)
}
