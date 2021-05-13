package dual

// Config configures the neural network
type Config struct {
	K            int  `json:"k"`             // number of filters
	SharedLayers int  `json:"shared_layers"` // number of shared residual blocks
	FC           int  `json:"fc"`            // fc layer width
	BatchSize    int  `json:"batch_size"`    // batch size
	Width        int  `json:"width"`         // board size width
	Height       int  `json:"height"`        // board size height
	Features     int  `json:"features"`      // feature counts
	ActionSpace  int  `json:"action_space"`  // action space
	FwdOnly      bool `json:"fwd_only"`      // is this a fwd only graph?
}

func DefaultConf(m, n, actionSpace int) Config {
	k := round((m * n) / 3)
	return Config{
		K:            k,
		SharedLayers: m,
		FC:           2 * k,
		BatchSize:    256,
		Width:        n,
		Height:       m,
		Features:     18,
		ActionSpace:  actionSpace,
	}
}

func (conf Config) IsValid() bool {
	return conf.K >= 1 &&
		conf.ActionSpace >= 3 &&
		// conf.SharedLayers >= conf.BoardSize &&
		conf.SharedLayers >= 0 &&
		conf.FC > 1 &&
		conf.BatchSize >= 1 &&
		// conf.ActionSpace >= conf.Width*conf.Height &&
		conf.Features > 0
}

func round(a int) int {
	n := a - 1
	n |= n >> 1
	n |= n >> 2
	n |= n >> 4
	n |= n >> 8
	n |= n >> 16
	n++

	lt := n / 2
	if (a - lt) < (n - a) {
		return lt
	}
	return n
}
