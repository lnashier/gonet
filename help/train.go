package help

import (
	"context"
	"fmt"
	"github.com/lnashier/gonet"
	"golang.org/x/sync/errgroup"
	"math"
	"time"
)

func Train(ctx context.Context, nn gonet.Network, epochs int, inputs, outputs [][]float64) {
	wg, trainCtx := errgroup.WithContext(ctx)

	trainingDone := make(chan struct{})
	currentEpoch := -1

	wg.Go(func() error {
		nn.Train(epochs, inputs, outputs, func(epoch int) bool {
			currentEpoch = epoch

			if currentEpoch%(epochs/10) == 0 {
				totalLoss := 0.0
				for i, input := range inputs {
					output := nn.Predict(input)
					for j := range output {
						totalLoss += math.Pow(outputs[i][j]-output[j], 2)
					}
				}
				averageLoss := totalLoss / float64(len(inputs))
				fmt.Printf("Epoch %04d, Loss: %f\n", currentEpoch, averageLoss)

				stats := nn.EpochStats(currentEpoch)
				if stats.Inputs != 0 {
					end := stats.End
					if end.IsZero() {
						end = time.Now()
					}
					fmt.Printf("Epoch:(%d) Inputs:(%d) Duration:(%v)\n", stats.ID, stats.Inputs, end.Sub(stats.Start))
				}
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
		ticker := time.NewTicker(5 * time.Second)
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

	fmt.Println("TrainingDuration", nn.TrainingDuration())
}
