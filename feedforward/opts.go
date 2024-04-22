package feedforward

type NetworkOpt func(*networkOpts)

type networkOpts struct {
	shapes               []int
	learningRate         float64
	activation           func(float64) float64
	activationDerivative func(float64) float64
}

var defaultNetworkOpts = networkOpts{
	learningRate: 0.1,
}

func (s *networkOpts) apply(opts []NetworkOpt) {
	for _, o := range opts {
		o(s)
	}
}

func Shapes(v []int) NetworkOpt {
	return func(s *networkOpts) {
		s.shapes = v
	}
}

func LearningRate(v float64) NetworkOpt {
	return func(s *networkOpts) {
		s.learningRate = v
	}
}

func Activation(v func(float64) float64) NetworkOpt {
	return func(s *networkOpts) {
		s.activation = v
	}
}

func ActivationDerivative(v func(float64) float64) NetworkOpt {
	return func(s *networkOpts) {
		s.activationDerivative = v
	}
}
