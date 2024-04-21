# Neural Networks

[![GoDoc](https://pkg.go.dev/badge/github.com/lnashier/goarc)](https://pkg.go.dev/github.com/lnashier/gonet)
[![Go Report Card](https://goreportcard.com/badge/github.com/lnashier/gonet)](https://goreportcard.com/report/github.com/lnashier/goarc)

## Examples

### Feedforward

[Examples](examples/feedforward/README.md)

### How to Build & Train

```go
// Start with defining shapes of your data and construct the network.
nn := feedforward.New(
    // We want to model 3 variables XOR function, that's the first shape
    // We want 4 nodes in first hidden layer, that's the second shape
    // There will be 1 output, that's the third shape.
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
