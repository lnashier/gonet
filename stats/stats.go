package stats

import (
	"sync"
	"time"
)

type Training struct {
	Start  time.Time
	End    time.Time
	Epochs sync.Map
}

type Epoch struct {
	ID     int
	Start  time.Time
	End    time.Time
	Inputs int
}
