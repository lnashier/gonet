package feedforward

import (
	"fmt"
	"github.com/lnashier/gonet/fns"
	"math/rand"
)

// Network is a basic implementation of a Multilayer Perceptron (MLP) a fully connected feedforward neural network.
type Network struct {
	inputSize           int
	hiddenSize          int
	outputSize          int
	weightsInputHidden  [][]float64
	biasesInputHidden   []float64
	weightsHiddenOutput [][]float64
	biasesHiddenOutput  []float64
	lr                  float64
}

// New initializes the weights and biases for the Network.
func New(inputSize, hiddenSize, outputSize int, lr float64) *Network {
	return &Network{
		inputSize:           inputSize,
		hiddenSize:          hiddenSize,
		outputSize:          outputSize,
		weightsInputHidden:  randomMat(inputSize, hiddenSize),
		biasesInputHidden:   randomArr(hiddenSize),
		weightsHiddenOutput: randomMat(hiddenSize, outputSize),
		biasesHiddenOutput:  randomArr(outputSize),
		lr:                  lr,
	}
}

func randomMat(rows, cols int) [][]float64 {
	weights := make([][]float64, rows)
	for i := range weights {
		weights[i] = randomArr(cols)
	}
	return weights
}

func randomArr(n int) []float64 {
	weights := make([]float64, n)
	for j := range weights {
		weights[j] = rand.Float64() - 0.5
	}
	return weights
}

func (nn *Network) forward(input []float64) ([]float64, []float64) {
	hiddenActivations := make([]float64, nn.hiddenSize)
	for i := range hiddenActivations {
		sum := nn.biasesInputHidden[i]
		for j := range input {
			sum += input[j] * nn.weightsInputHidden[j][i]
		}
		hiddenActivations[i] = fns.Sigmoid(sum)
	}

	output := make([]float64, nn.outputSize)
	for i := range output {
		sum := nn.biasesHiddenOutput[i]
		for j := range hiddenActivations {
			sum += hiddenActivations[j] * nn.weightsHiddenOutput[j][i]
		}
		output[i] = fns.Sigmoid(sum)
	}

	return output, hiddenActivations
}

func (nn *Network) backward(input []float64, targetOutput []float64) {
	output, hiddenActivations := nn.forward(input)

	outputError := make([]float64, nn.outputSize)
	for i := range outputError {
		outputError[i] = targetOutput[i] - output[i]
	}

	outputDelta := make([]float64, nn.outputSize)
	for i := range outputDelta {
		outputDelta[i] = outputError[i] * fns.SigmoidDerivative(output[i])
	}

	hiddenError := make([]float64, nn.hiddenSize)
	for i := range hiddenError {
		for j := range nn.weightsHiddenOutput[i] {
			hiddenError[i] += outputDelta[j] * nn.weightsHiddenOutput[i][j]
		}
	}

	hiddenDelta := make([]float64, nn.hiddenSize)
	for i := range hiddenDelta {
		hiddenDelta[i] = hiddenError[i] * fns.SigmoidDerivative(hiddenActivations[i])
	}

	for i := range nn.weightsHiddenOutput {
		for j := range nn.weightsHiddenOutput[i] {
			nn.weightsHiddenOutput[i][j] += nn.lr * hiddenActivations[i] * outputDelta[j]
		}
	}

	for i := range nn.biasesHiddenOutput {
		nn.biasesHiddenOutput[i] += nn.lr * outputDelta[i]
	}

	for i := range nn.weightsInputHidden {
		for j := range nn.weightsInputHidden[i] {
			nn.weightsInputHidden[i][j] += nn.lr * input[i] * hiddenDelta[j]
		}
	}

	for i := range nn.biasesInputHidden {
		nn.biasesInputHidden[i] += nn.lr * hiddenDelta[i]
	}
}

func (nn *Network) Predict(input []float64) []float64 {
	output, _ := nn.forward(input)
	return output
}

func (nn *Network) Train(inputs [][]float64, outputs [][]float64, epochs int, callback func(int)) {
	for epoch := range epochs {
		for i, input := range inputs {
			nn.backward(input, outputs[i])
		}
		callback(epoch)
	}
}

func (nn *Network) String() string {
	return fmt.Sprintf(
		"Input Size: %d\n"+
			"Hidden Size: %d\n"+
			"Output Size: %d\n"+
			"Weights Input to Hidden: %v\n"+
			"Biases Input to Hidden: %v\n"+
			"Weights Hidden to Output: %v\n"+
			"Biases Hidden to Output: %v\n",
		nn.inputSize,
		nn.hiddenSize,
		nn.outputSize,
		nn.weightsInputHidden,
		nn.biasesInputHidden,
		nn.weightsHiddenOutput,
		nn.biasesHiddenOutput,
	)
}
