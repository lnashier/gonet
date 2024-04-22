package gonet

import (
	"github.com/lnashier/gonet/stats"
	"io"
	"time"
)

type Network interface {
	Train(epochs int, inputs, outputs [][]float64, callback func(int) bool)
	Predict(inputs []float64) []float64
	TrainingDuration() time.Duration
	EpochStats(epoch int) stats.Epoch
	Save(w io.Writer) error
	String() string
}
