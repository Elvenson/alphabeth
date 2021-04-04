package agogo

import (
	"io"

	dual "github.com/alphabeth/dualnet"
	"github.com/alphabeth/game"
	"github.com/alphabeth/mcts"
)

// Config for the AZ structure.
// It holds attributes that impacts the MCTS and the Neural Network
// as well as object that facilitates the interactions with the end-user (eg: OutputEncoder).
type Config struct {
	Name            string
	NNConf          dual.Config
	MCTSConf        mcts.Config
	UpdateThreshold float64
	MaxExamples     int // maximum number of examples

	// extensions
	Encoder   GameEncoder
	Augmenter Augmenter
}

// GameEncoder encodes a game state as a slice of floats
type GameEncoder func(a game.State) []float32

// Augmenter takes an example, and creates more examples from it.
type Augmenter func(a Example) []Example

// Example is a representation of an example.
type Example struct {
	Board  []float32
	Policy []float32
	Value  float32
}

// Dualer is an interface for anything that allows getting out a *Dual.
//
// Its sole purpose is to form a monoid-ish data structure for Agent.NN
type Dualer interface {
	Dual() *dual.Dual
}

// Inferer is anything that can infer given an input.
type Inferer interface {
	Infer(a []float32) (policy []float32, value float32, err error)
	io.Closer
}

// ExecLogger is anything that can return the execution log.
type ExecLogger interface {
	ExecLog() string
}