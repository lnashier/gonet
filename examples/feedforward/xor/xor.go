package xor

import (
	"fmt"
	"github.com/lnashier/gonet"
	"github.com/lnashier/gonet/fns"
	"math"
	"os"
)

var trainingData = [][]float64{
	{0, 0},
	{0, 1},
	{1, 0},
	{1, 1},
}

var targetOutputs = [][]float64{
	{0},
	{1},
	{1},
	{0},
}

func getModel(name string) (*gonet.FeedforwardNetwork, bool) {
	model, _ := os.ReadFile(name)
	return gonet.Feedforward(
		gonet.LoadFrom(model),
		gonet.InputSize(2),
		gonet.HiddenSize(4),
		gonet.OutputSize(1),
		gonet.LearningRate(0.1),
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

func train(nn *gonet.FeedforwardNetwork) {
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
}

// Build trains a XOR function
func Build() {
	nn, loaded := getModel("bin/xor.gob")
	if !loaded {
		train(nn)
		err := saveModel("bin/xor.gob", nn)
		if err != nil {
			panic(err)
		}
	}

	for _, input := range trainingData {
		output := nn.Predict(input)
		fmt.Println(input, "->", output)
	}
}
