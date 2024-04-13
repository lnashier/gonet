package gonet

type Network interface {
	Train(training [][]float64, target [][]float64, epochs int)
	Predict(input []float64) []float64
	String() string
}
