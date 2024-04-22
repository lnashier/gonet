package help

import (
	"github.com/lnashier/gonet"
	"os"
)

func Save(name string, nn gonet.Network) error {
	file, err := os.Create(name)
	if err != nil {
		panic(err)
	}
	defer file.Close()
	return nn.Save(file)
}
