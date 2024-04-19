package fns

import (
	"math"
	"math/rand"
)

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func SigmoidDerivative(x float64) float64 {
	return x * (1 - x)
}

func ReLU(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

func ReLUDerivative(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

func Argmax(values []float64) int {
	maxValue := math.Inf(-1)
	maxIndex := -1
	for i, v := range values {
		if v > maxValue {
			maxValue = v
			maxIndex = i
		}
	}
	return maxIndex
}

func MeanSquaredError(predicted, target []float64) float64 {
	if len(predicted) != len(target) {
		panic("predicted and target output sizes must match")
	}
	var sum float64
	for i := range predicted {
		diff := predicted[i] - target[i]
		sum += diff * diff
	}
	return sum / float64(len(predicted))
}

func RandomMat(rows, cols int) [][]float64 {
	weights := make([][]float64, rows)
	for i := range weights {
		weights[i] = RandomVector(cols)
	}
	return weights
}

func RandomVector(n int) []float64 {
	weights := make([]float64, n)
	for j := range weights {
		weights[j] = rand.Float64() - 0.5
	}
	return weights
}
