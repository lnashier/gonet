package help

import (
	"github.com/lnashier/gonet/feedforward"
	"os"
)

func LoadFeedforward(name string, opt ...feedforward.NetworkOpt) (*feedforward.Network, error) {
	model, err := os.Open(name)
	if err != nil {
		return nil, err
	}
	defer model.Close()
	return feedforward.Load(model, opt...)
}
