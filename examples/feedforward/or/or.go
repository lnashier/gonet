package or

import (
	"fmt"
	"github.com/lnashier/gonet"
	"github.com/lnashier/gonet/fns"
	"math"
)

// Build creates and train an OR function
func Build() {
	inputSize := 2
	hiddenSize := 4
	outputSize := 1

	trainingData := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	targetOutputs := [][]float64{
		{0},
		{1},
		{1},
		{1},
	}

	nn := gonet.Feedforward(
		gonet.InputSize(inputSize),
		gonet.HiddenSize(hiddenSize),
		gonet.OutputSize(outputSize),
		gonet.Activation(fns.Sigmoid),
		gonet.ActivationDerivative(fns.SigmoidDerivative),
		gonet.LearningRate(1),
	)

	epochs := 10000

	nn.Train(epochs, trainingData, targetOutputs, func(epoch int) bool {
		if epoch%(epochs/10) == 0 {
			totalLoss := 0.0
			for i, input := range trainingData {
				output := nn.Predict(input)
				for j := range output {
					totalLoss += math.Pow(targetOutputs[i][j]-output[j], 2)
				}
			}
			averageLoss := totalLoss / float64(len(trainingData))
			fmt.Printf("Epoch %04d, Loss: %f\n", epoch, averageLoss)
		}
		return true
	})

	fmt.Println("TrainingDuration", nn.TrainingDuration())

	for _, input := range trainingData {
		output := nn.Predict(input)
		fmt.Println(input, "->", output)
	}
}
