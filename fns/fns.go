package fns

import (
	"math"
	"math/rand"
)

func Sigmoid(x float64) float64 {
	// sigmoid(x) = 1 / (1 + exp(-x))
	return 1 / (1 + math.Exp(-x))
}

func SigmoidDerivative(x float64) float64 {
	// sigmoid'(j) = sigmoid(j) * (1 - sigmoid(j))
	// here x = sigmoid(j)
	return x * (1 - x)
}

func ReLU(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

func ReLUDerivative(x float64) float64 {
	return ReLU(x)
}

func Tanh(x float64) float64 {
	return math.Tanh(x)
}

func TanhDerivative(x float64) float64 {
	// tanh'(j) = 1 - tanh^2(j)
	// here x = math.Tanh(j)
	return 1 - math.Pow(x, 2)
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

func RandomMatrix(rows, cols int) [][]float64 {
	mat := make([][]float64, rows)
	for i := range mat {
		mat[i] = RandomVector(cols)
	}
	return mat
}

func RandomVector(n int) []float64 {
	vec := make([]float64, n)
	for j := range vec {
		vec[j] = rand.Float64() - 0.5
	}
	return vec
}

// FnVec applies given function f to each element of the vector.
func FnVec(vec []float64, f func(float64) float64) []float64 {
	result := make([]float64, len(vec))
	for i, v := range vec {
		result[i] = f(v)
	}
	return result
}

// Scalar multiplies a vector by a scalar.
func Scalar(vec []float64, scalar float64) []float64 {
	result := make([]float64, len(vec))
	for i, v := range vec {
		result[i] = v * scalar
	}
	return result
}

// Dot computes the dot product of two matrices.
func Dot(mat1 [][]float64, mat2 [][]float64) [][]float64 {
	if len(mat1[0]) != len(mat2) {
		panic("can't multiply matrices")
	}

	result := make([][]float64, len(mat1))

	for i := 0; i < len(mat1); i++ {
		result[i] = make([]float64, len(mat2[0]))
		for j := 0; j < len(mat2[0]); j++ {
			dot := 0.0
			for k := 0; k < len(mat2); k++ {
				dot += mat1[i][k] * mat2[k][j]
			}
			result[i][j] = dot
		}
	}

	return result
}

// Transpose computes the transpose of a matrix.
func Transpose(mat [][]float64) [][]float64 {
	result := make([][]float64, len(mat[0]))
	for i := range result {
		result[i] = make([]float64, len(mat))
		for j := range result[i] {
			result[i][j] = mat[j][i]
		}
	}
	return result
}

// AddVec adds two vectors element-wise.
func AddVec(vec1, vec2 []float64) []float64 {
	result := make([]float64, len(vec1))
	for i := range result {
		result[i] = vec1[i] + vec2[i]
	}
	return result
}

// SubtractVec subtracts one vector from another element-wise.
func SubtractVec(vec1, vec2 []float64) []float64 {
	result := make([]float64, len(vec1))
	for i := range result {
		result[i] = vec1[i] - vec2[i]
	}
	return result
}

// AddMat adds two matrices element-wise.
func AddMat(mat1, mat2 [][]float64) [][]float64 {
	result := make([][]float64, len(mat1))
	for i := range result {
		result[i] = make([]float64, len(mat1[i]))
		for j := range result[i] {
			result[i][j] = mat1[i][j] + mat2[i][j]
		}
	}
	return result
}

// SubtractMat subtracts one matrix from another element-wise.
func SubtractMat(mat1, mat2 [][]float64) [][]float64 {
	result := make([][]float64, len(mat1))
	for i := range result {
		result[i] = make([]float64, len(mat1[i]))
		for j := range result[i] {
			result[i][j] = mat1[i][j] - mat2[i][j]
		}
	}
	return result
}

func MeanSquaredError(predictions, targets [][]float64) float64 {
	totalLoss := 0.0
	for i, prediction := range predictions {
		target := targets[i]
		for j := range prediction {
			loss := target[j] - prediction[j]
			totalLoss += loss * loss
		}
	}
	return totalLoss / float64(len(predictions))
}

func LogLoss(predictions, targets [][]float64) float64 {
	totalLoss := 0.0
	for i, prediction := range predictions {
		target := targets[i]
		for j := range prediction {
			totalLoss += -((target[j] * math.Log(prediction[j])) + ((1 - target[j]) * math.Log(1-prediction[j])))
		}
	}
	return totalLoss / float64(len(predictions))
}

func BinaryLogLoss(predictions, targets [][]float64) float64 {
	totalLoss := 0.0
	for i, prediction := range predictions {
		target := targets[i]
		totalLoss += -((target[0] * math.Log(prediction[0])) + ((1 - target[0]) * math.Log(1-prediction[0])))
	}
	return totalLoss / float64(len(predictions))
}
