# Neural Networks

[![GoDoc](https://pkg.go.dev/badge/github.com/lnashier/goarc)](https://pkg.go.dev/github.com/lnashier/gonet)
[![Go Report Card](https://goreportcard.com/badge/github.com/lnashier/gonet)](https://goreportcard.com/report/github.com/lnashier/goarc)

## Examples

[Feedforward Examples](examples/feedforward/README.md)

## How to Build & Train

Start with defining shapes of your data. For example:

```go
// We want to model 3 variables XOR function
inputSize := 3
// Output is a single value 0 or 1
outputSize := 1
// Don't worry about this too much at this moment
// It is simply saying how many nodes in first hidden layer
// At the moment, only single layer FFN is supported
hiddenSize := 4
```

Next, let's construct neural network

```go
nn := gonet.Feedforward(
    gonet.InputSize(inputSize), // defined above
    gonet.HiddenSize(hiddenSize), // defined above
    gonet.OutputSize(outputSize), // defined above
    // Sigmoid function squashes the output to the range [0, 1], that makes it suitable for binary classification.
    gonet.Activation(fns.Sigmoid), // OK; this is chosen based on prior knowledge of training data 
    gonet.ActivationDerivative(fns.SigmoidDerivative), // For backpropagation 
    gonet.LearningRate(0.1), // Learning rate, choose wisely
)
```
Let's print shapes
```go
fmt.Println(nn.String())
// Input Size: 3
// Hidden Size: 4
// Output Size: 1
// Learning Rate: 0.1000
```

Let's define training data

```go
trainingInputs := [][]float64{
    {0, 0, 0},
    {0, 0, 1},
    {0, 1, 0},
    {0, 1, 1},
    {1, 0, 0},
    {1, 0, 1},
    {1, 1, 0},
    {1, 1, 1},
}
targetOutputs := [][]float64{
    {0},
    {1},
    {1},
    {1},
    {1},
    {1},
    {1},
    {0},
}
```

Train/fit the function

```go
help.Train(context.TODO(), nn, 10000, trainingInputs, targetOutputs)
```

Lets predict some values

```go
fmt.Println(nn.Predict([]float64{0, 0, 0})) // 0.04805021974569349
fmt.Println(nn.Predict([]float64{1, 1, 0})) // 0.9570106865711672
fmt.Println(nn.Predict([]float64{1, 1, 1})) // 0.0679005862480516
```
