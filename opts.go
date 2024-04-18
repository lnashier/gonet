package gonet

type NetworkOpt func(*networkOpts)

type networkOpts struct {
	inputSize            int
	hiddenSize           int
	outputSize           int
	learningRate         float64
	activation           func(float64) float64
	activationDerivative func(float64) float64
}

var defaultNetworkOpts = networkOpts{}

func (s *networkOpts) apply(opts []NetworkOpt) {
	for _, o := range opts {
		o(s)
	}
}

func InputSize(v int) NetworkOpt {
	return func(s *networkOpts) {
		s.inputSize = v
	}
}

func HiddenSize(v int) NetworkOpt {
	return func(s *networkOpts) {
		s.hiddenSize = v
	}
}

func OutputSize(v int) NetworkOpt {
	return func(s *networkOpts) {
		s.outputSize = v
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
