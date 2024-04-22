package stats

import "time"

type Training struct {
	Start  time.Time
	End    time.Time
	Epochs map[int]*Epoch
}

type Epoch struct {
	ID     int
	Start  time.Time
	End    time.Time
	Inputs int
}
