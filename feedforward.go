package gonet

import (
	"fmt"
	"github.com/lnashier/gonet/fns"
)

// FeedforwardNetwork is a basic implementation of a Multilayer Perceptron (MLP) a fully connected feedforward neural network.
type FeedforwardNetwork struct {
	inputSize           int
	hiddenSize          int
	outputSize          int
	weightsInputHidden  [][]float64
	biasesHidden        []float64
	weightsHiddenOutput [][]float64
	biasesOutput        []float64
	lr                  float64
	af                  func(float64) float64
	fd                  func(float64) float64
}

// Feedforward initializes the weights and biases for the FeedforwardNetwork.
func Feedforward(opt ...NetworkOpt) *FeedforwardNetwork {
	opts := defaultNetworkOpts
	opts.apply(opt)

	return &FeedforwardNetwork{
		inputSize:           opts.inputSize,
		hiddenSize:          opts.hiddenSize,
		outputSize:          opts.outputSize,
		weightsInputHidden:  fns.RandomMat(opts.inputSize, opts.hiddenSize),
		biasesHidden:        fns.RandomVector(opts.hiddenSize),
		weightsHiddenOutput: fns.RandomMat(opts.hiddenSize, opts.outputSize),
		biasesOutput:        fns.RandomVector(opts.outputSize),
		lr:                  opts.learningRate,
		af:                  opts.activation,
		fd:                  opts.activationDerivative,
	}
}

func (nn *FeedforwardNetwork) forward(input []float64) ([]float64, []float64) {
	hiddenActivations := make([]float64, nn.hiddenSize)
	for i := range hiddenActivations {
		sum := nn.biasesHidden[i]
		for j := range input {
			sum += input[j] * nn.weightsInputHidden[j][i]
		}
		hiddenActivations[i] = nn.af(sum)
	}

	output := make([]float64, nn.outputSize)
	for i := range output {
		sum := nn.biasesOutput[i]
		for j := range hiddenActivations {
			sum += hiddenActivations[j] * nn.weightsHiddenOutput[j][i]
		}
		output[i] = nn.af(sum)
	}

	return hiddenActivations, output
}

func (nn *FeedforwardNetwork) backward(input []float64, targetOutput []float64) {
	hiddenActivations, output := nn.forward(input)

	outputError := make([]float64, nn.outputSize)
	for i := range outputError {
		outputError[i] = targetOutput[i] - output[i]
	}

	outputDelta := make([]float64, nn.outputSize)
	for i := range outputDelta {
		outputDelta[i] = outputError[i] * nn.fd(output[i])
	}

	hiddenError := make([]float64, nn.hiddenSize)
	for i := range hiddenError {
		for j := range nn.weightsHiddenOutput[i] {
			hiddenError[i] += outputDelta[j] * nn.weightsHiddenOutput[i][j]
		}
	}

	hiddenDelta := make([]float64, nn.hiddenSize)
	for i := range hiddenDelta {
		hiddenDelta[i] = hiddenError[i] * nn.fd(hiddenActivations[i])
	}

	for i := range nn.weightsHiddenOutput {
		for j := range nn.weightsHiddenOutput[i] {
			nn.weightsHiddenOutput[i][j] += nn.lr * hiddenActivations[i] * outputDelta[j]
		}
	}

	for i := range nn.biasesOutput {
		nn.biasesOutput[i] += nn.lr * outputDelta[i]
	}

	for i := range nn.weightsInputHidden {
		for j := range nn.weightsInputHidden[i] {
			nn.weightsInputHidden[i][j] += nn.lr * input[i] * hiddenDelta[j]
		}
	}

	for i := range nn.biasesHidden {
		nn.biasesHidden[i] += nn.lr * hiddenDelta[i]
	}
}

func (nn *FeedforwardNetwork) Predict(input []float64) []float64 {
	_, output := nn.forward(input)
	return output
}

func (nn *FeedforwardNetwork) Train(inputs [][]float64, outputs [][]float64, epochs int, callback func(int)) {
	for epoch := range epochs {
		for i, input := range inputs {
			nn.backward(input, outputs[i])
		}
		callback(epoch)
	}
}

func (nn *FeedforwardNetwork) String() string {
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
		nn.biasesHidden,
		nn.weightsHiddenOutput,
		nn.biasesOutput,
	)
}
