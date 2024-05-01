package help

import (
	"context"
	"fmt"
	"github.com/lnashier/gonet"
	"github.com/lnashier/gonet/fns"
	"golang.org/x/sync/errgroup"
	"time"
)

func Train(ctx context.Context, nn gonet.Network, epochs int, inputs, targets [][]float64, opt ...TrainingOpt) {
	opts := defaultTrainingOpts
	opts.apply(opt)

	wg, trainCtx := errgroup.WithContext(ctx)

	trainingDone := make(chan struct{})
	currentEpoch := -1

	wg.Go(func() error {
		nn.Train(epochs, inputs, targets, func(epoch int) bool {
			currentEpoch = epoch

			if currentEpoch%(epochs/10) == 0 {
				stats := nn.EpochStats(currentEpoch)
				if stats.Inputs != 0 {
					end := stats.End
					if end.IsZero() {
						end = time.Now()
					}
					fmt.Printf("Epoch:(%d) Inputs:(%d) Duration:(%v)\n", stats.ID, stats.Inputs, end.Sub(stats.Start))
				}

				predictions := make([][]float64, len(inputs))
				for i, input := range inputs {
					predictions[i] = nn.Predict(input)
				}

				fmt.Printf("Epoch %04d, Loss: %f\n", currentEpoch, opts.lossFunc(predictions, targets))
			}

			contTraining := true
			select {
			case <-trainCtx.Done():
				contTraining = false
			default:
			}
			return contTraining
		})
		trainingDone <- struct{}{}
		return nil
	})

	wg.Go(func() error {
		ticker := time.NewTicker(opts.echoStatsEvery)
		for {
			select {
			case <-trainingDone:
				return nil
			case <-ticker.C:
				stats := nn.EpochStats(currentEpoch + 1)
				if stats.Inputs != 0 {
					end := stats.End
					if end.IsZero() {
						end = time.Now()
					}
					fmt.Printf("Epoch:(%d) Inputs:(%d) Duration:(%v)\n", stats.ID, stats.Inputs, end.Sub(stats.Start))
				}
			}
		}
	})

	err := wg.Wait()
	if err != nil {
		panic(err)
	}

	fmt.Println("Training Duration", nn.TrainingDuration())
}

type TrainingOpt func(*trainingOpts)

type trainingOpts struct {
	echoStatsEvery time.Duration
	lossFunc       func(predictions, targets [][]float64) float64
}

var defaultTrainingOpts = trainingOpts{
	echoStatsEvery: 5 * time.Second,
	lossFunc:       fns.MeanSquaredError, // Default loss function MSE
}

func (s *trainingOpts) apply(opts []TrainingOpt) {
	for _, o := range opts {
		o(s)
	}
}

func EchoStatsEvery(v time.Duration) TrainingOpt {
	return func(s *trainingOpts) {
		s.echoStatsEvery = v
	}
}

// LossFunc sets the loss function for the network being trained.
func LossFunc(v func(predictions, targets [][]float64) float64) TrainingOpt {
	return func(s *trainingOpts) {
		s.lossFunc = v
	}
}
