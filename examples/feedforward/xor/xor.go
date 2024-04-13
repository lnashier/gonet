package xor

import (
	"fmt"
	"github.com/lnashier/gonet/feedforward"
	"math"
)

// Build creates and train a XOR function
func Build() {
	inputSize := 2
	hiddenSize := 4
	outputSize := 1
	lr := 0.1

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
		{0},
	}

	nn := feedforward.New(inputSize, hiddenSize, outputSize, lr)

	nn.Train(trainingData, targetOutputs, 10000000, func(epoch int) {
		if epoch%1000000 == 0 {
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
	})

	fmt.Println(nn.String())

	for _, input := range trainingData {
		output := nn.Predict(input)
		fmt.Println(input, "->", output)
	}
}
