# Neural Networks

[![GoDoc](https://pkg.go.dev/badge/github.com/lnashier/goarc)](https://pkg.go.dev/github.com/lnashier/gonet)
[![Go Report Card](https://goreportcard.com/badge/github.com/lnashier/gonet)](https://goreportcard.com/report/github.com/lnashier/goarc)

## Examples

### Feedforward

[Examples](examples/feedforward/README.md)

### How to Build & Train

```go
// Start by defining shapes of your data and construct the network.

nn := feedforward.New(
    // We want to model 3 variables XOR function, that's the first shape.
    // We want 4 nodes in first hidden layer, that's the second shape.
    // There will be 1 output (0 or 1), that's the third shape.
    feedforward.Shapes([]int{3, 4, 1}),

    // This is chosen based on prior knowledge of training data.
    // Sigmoid function squashes the output to the range [0, 1],
    // that makes it suitable for binary classification.
    feedforward.Activation(fns.Sigmoid),

    // backpropagation
    feedforward.ActivationDerivative(fns.SigmoidDerivative),

    // Learning rate, choose wisely
    feedforward.LearningRate(0.1),
)

// Print shapes

fmt.Println(nn.String())
// Shapes: [3 4 1]
// Hidden Layers: 1

// Define training data

inputs := [][]float64{
    {0, 0, 0},
    {0, 0, 1},
    {0, 1, 0},
    {0, 1, 1},
    {1, 0, 0},
    {1, 0, 1},
    {1, 1, 0},
    {1, 1, 1},
}

targets := [][]float64{
    {0},
    {1},
    {1},
    {1},
    {1},
    {1},
    {1},
    {0},
}

// Train/fit the function

help.Train(context.TODO(), nn, 100000, inputs, targets)
// Epoch 0000, Loss: 0.214975
// Epoch:(0) Inputs:(8) Duration:(72.166µs)
// Epoch 10000, Loss: 0.004950
// Epoch:(10000) Inputs:(8) Duration:(2.542µs)
// Epoch 20000, Loss: 0.001701
// Epoch:(20000) Inputs:(8) Duration:(2µs)
// Epoch 30000, Loss: 0.001014
// Epoch:(30000) Inputs:(8) Duration:(1.834µs)
// ...
// ...

// Let's predict

fmt.Println(nn.Predict([]float64{0, 0, 0})) // [0.02768982884099321]
fmt.Println(nn.Predict([]float64{1, 1, 0})) // [0.9965961389721709]
fmt.Println(nn.Predict([]float64{1, 1, 1})) // [0.012035375150277857]
```

### How to Save & Resume

```go
// Saving a network to disk allows for later loading to resume training and/or prediction.

// To save the Network, you would invoke Save() on it and provide an io.Writer as an argument.
if err := nn.Save(w); err != nil {
    // do something with error
}

// Alternatively, you can call help.Save, specifying a file name (which can include a file path).
if err := help.Save("bin/my-model", nn); err != nil {
    // do something with error
}
```

```go
// A network can be loaded from disk to resume training and/or to predict.

// To load a previously saved network, you would call feedforward.Load and pass in an io.Reader. 
nn, err := feedforward.Load(
    r, // io.Reader
    feedforward.Activation(fns.Sigmoid), // set for your network
    // REQUIRED if resuming network training
    feedforward.ActivationDerivative(fns.SigmoidDerivative), // set for your network
    feedforward.LearningRate(0.01), // set for your network
)
if err != nil {
    // do something with error
}

// Alternatively, you can call help.LoadFeedforward, providing a file name (which can be a file path).
nn, err := help.LoadFeedforward(
    "bin/my-model",
    feedforward.Activation(fns.Sigmoid), // set for your network
    // REQUIRED if resuming network training
    feedforward.ActivationDerivative(fns.SigmoidDerivative), // set for your network
    feedforward.LearningRate(0.01), // set for your network
)
if err != nil {
    // do something with error
}
```
