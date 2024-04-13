package main

import (
	"context"
	"feedforward/mnist"
	"feedforward/or"
	"feedforward/xor"
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
					or.Build()
				case "xor":
					xor.Build()
				case "mnist":
					mnist.Build(args[1:])
				}
				return nil
			})
			return nil
		}),
	))
}
