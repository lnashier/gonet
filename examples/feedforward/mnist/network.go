package mnist

import (
	"fmt"
	"github.com/lnashier/gonet/feedforward"
	"github.com/lnashier/gonet/fns"
	"math"
)

func Build(args []string) {
	inputSize := 28 * 28
	hiddenSize := 128
	outputSize := 10
	lr := 5.0

	trainingInputs, trainingOutputs, err := trainingData(outputSize, args[0], args[1])
	if err != nil {
		panic(err)
	}

	nn := feedforward.New(inputSize, hiddenSize, outputSize, lr)

	nn.Train(trainingInputs, trainingOutputs, 1, func(epoch int) {
		if epoch%100 == 0 {
			totalLoss := 0.0
			for i, input := range trainingInputs {
				output := nn.Predict(input)
				for j := range output {
					totalLoss += math.Pow(trainingOutputs[i][j]-output[j], 2)
				}
			}
			averageLoss := totalLoss / float64(len(trainingInputs))
			fmt.Printf("Epoch %04d, Loss: %f\n", epoch, averageLoss)
		}
	})

	unseenInputs, unseenOutputs, err := readData(args[2], args[3])
	if err != nil {
		panic(err)
	}

	var correctPredictions int
	for i, input := range unseenInputs {
		prediction := nn.Predict(flattenImage(input))
		output := fns.Argmax(prediction)
		if output == int(unseenOutputs[i]) {
			err = saveImage(input, fmt.Sprintf("bin/correct/image-p%d-r%d.png", output, unseenOutputs[i]))
			if err != nil {
				panic(err)
			}
			correctPredictions++
		} else {
			err = saveImage(input, fmt.Sprintf("bin/wrong/image-p%d-r%d.png", output, unseenOutputs[i]))
			if err != nil {
				panic(err)
			}
		}
	}

	accuracy := float64(correctPredictions) / float64(len(unseenOutputs))
	fmt.Printf("Unseen Outputs: %d, Correct Predictions: %d, Accuracy: %.2f%%\n", len(unseenOutputs), correctPredictions, accuracy*100)
}
