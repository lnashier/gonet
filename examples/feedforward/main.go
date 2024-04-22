package main

import (
	"context"
	"feedforward/mnist"
	"feedforward/or"
	"feedforward/sine"
	"feedforward/xor"
	"feedforward/xor3"
	"fmt"
	"github.com/lnashier/goarc"
	goarccli "github.com/lnashier/goarc/cli"
)

func main() {
	goarc.Up(goarccli.NewService(
		goarccli.ServiceName("Examples"),
		goarccli.App(func(svc *goarccli.Service) error {
			svc.Register("build", func(ctx context.Context, args []string) error {
				switch args[0] {
				case "or":
					or.Build(ctx)
				case "xor":
					xor.Build(ctx, args[1:])
				case "xor3":
					xor3.Build(ctx)
				case "sine":
					sine.Build(ctx, args[1:])
				case "mnist":
					mnist.Build(ctx, args[1:])
				default:
					return fmt.Errorf("model not found: %s", args[0])
				}
				return nil
			})
			return nil
		}),
	))
}
