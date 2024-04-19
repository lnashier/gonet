package mnist

import (
	"context"
	"fmt"
	"github.com/lnashier/gonet"
	"github.com/lnashier/gonet/fns"
	"golang.org/x/sync/errgroup"
	"math"
	"os"
	"time"
)

func getModel(name string) (*gonet.FeedforwardNetwork, bool) {
	model, _ := os.ReadFile(name)
	return gonet.Feedforward(
		gonet.LoadFrom(model),
		gonet.InputSize(28*28),
		gonet.HiddenSize(128),
		gonet.OutputSize(10),
		gonet.LearningRate(1),
		gonet.Activation(fns.Sigmoid),
		gonet.ActivationDerivative(fns.SigmoidDerivative),
	), len(model) > 0
}

func saveModel(name string, nn *gonet.FeedforwardNetwork) error {
	file, err := os.Create(name)
	if err != nil {
		panic(err)
	}
	defer file.Close()
	return nn.Save(file)
}

func train(ctx context.Context, nn *gonet.FeedforwardNetwork, inputs, outputs [][]float64) {
	wg, trainCtx := errgroup.WithContext(ctx)

	trainingDone := make(chan struct{})
	currentEpoch := -1
	epochs := 10

	wg.Go(func() error {
		nn.Train(epochs, inputs, outputs, func(epoch int) bool {
			currentEpoch = epoch

			stats := nn.EpochStats(currentEpoch)
			if stats.Inputs != 0 {
				end := stats.End
				if end.IsZero() {
					end = time.Now()
				}
				fmt.Printf(
					"Epoch:(%d) Inputs:(%d) Duration:(%v) Forward:(%d)(%v) Backward:(%d)(%v)\n",
					stats.ID,
					stats.Inputs,
					end.Sub(stats.Start),
					stats.Forward.Count,
					stats.Forward.Duration,
					stats.Backward.Count,
					stats.Backward.Duration,
				)
			}

			if epoch%(epochs/10) == 0 {
				totalLoss := 0.0
				for i, input := range inputs {
					output := nn.Predict(input)
					for j := range output {
						totalLoss += math.Pow(outputs[i][j]-output[j], 2)
					}
				}
				averageLoss := totalLoss / float64(len(inputs))
				fmt.Printf("Epoch %04d, Loss: %f\n", epoch, averageLoss)
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
					fmt.Printf(
						"Epoch:(%d) Inputs:(%d) Duration:(%v) Forward:(%d)(%v) Backward:(%d)(%v)\n",
						stats.ID,
						stats.Inputs,
						end.Sub(stats.Start),
						stats.Forward.Count,
						stats.Forward.Duration,
						stats.Backward.Count,
						stats.Backward.Duration,
					)
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

func test(ctx context.Context, nn *gonet.FeedforwardNetwork, inputs [][][]uint8, outputs []uint8) {
	var correctPredictions int
	var predictions int

	defer func() {
		accuracy := float64(correctPredictions) / float64(predictions)
		fmt.Printf("Total Predictions: %d, Correct Predictions: %d, Accuracy: %.2f%%\n", predictions, correctPredictions, accuracy*100)
	}()

	for i, input := range inputs {
		select {
		case <-ctx.Done():
			return
		default:
		}
		prediction := nn.Predict(flattenImage(input))
		predictions++
		output := fns.Argmax(prediction)
		if output == int(outputs[i]) {
			err := saveImage(input, fmt.Sprintf("bin/correct/image-p%d-r%d.png", output, outputs[i]))
			if err != nil {
				panic(err)
			}
			correctPredictions++
		} else {
			err := saveImage(input, fmt.Sprintf("bin/wrong/image-p%d-r%d.png", output, outputs[i]))
			if err != nil {
				panic(err)
			}
		}
	}
}

func Build(ctx context.Context, args []string) {
	nn, loaded := getModel("bin/mnist.gob")
	if !loaded {
		trainingInputs, trainingOutputs, err := trainingData(10, args[0], args[1])
		if err != nil {
			panic(err)
		}
		train(ctx, nn, trainingInputs, trainingOutputs)

		err = saveModel("bin/mnist.gob", nn)
		if err != nil {
			panic(err)
		}
	}

	unseenInputs, unseenOutputs, err := readData(args[2], args[3])
	if err != nil {
		panic(err)
	}
	test(ctx, nn, unseenInputs, unseenOutputs)
}
