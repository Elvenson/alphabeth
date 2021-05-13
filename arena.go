package agogo

import (
	"runtime"

	"github.com/alphabeth/game"
	"github.com/alphabeth/mcts"
	"github.com/chewxy/math32"
	"github.com/notnil/chess"
)

// Arena represents a game arena
// Arena fulfils the interface game.MetaState
type Arena struct {
	game         game.State
	CurrentAgent *Agent

	// state
	conf mcts.Config

	// only relevant to training
	name string
}

// MakeArena makes an arena given a game.
func MakeArena(g game.State, a Dualer, conf mcts.Config, enc GameEncoder, name string) Arena {
	CurrentAgent := &Agent{
		NN:   a.Dual(),
		Enc:  enc,
		name: "current agent",
	}
	CurrentAgent.MCTS = mcts.New(g, conf, CurrentAgent)

	return Arena{
		game:         g,
		CurrentAgent: CurrentAgent,
		conf:         conf,
		name:         name,
	}
}

// SelfPlay lets the agent to generate training data by playing with itself.
func (a *Arena) SelfPlay() (examples []Example, err error) {
	if err := a.CurrentAgent.SwitchToInference(); err != nil {
		return nil, err
	}

	var winner chess.Color
	var ended bool
	for ended, winner = a.game.Ended(); !ended; ended, winner = a.game.Ended() {
		best, err := a.CurrentAgent.Search(a.game)
		if err != nil {
			return nil, err
		}
		if best == game.ResignMove {
			break
		}

		boards := a.CurrentAgent.Enc(a.game)
		policies, err := a.CurrentAgent.MCTS.Policies()
		if err != nil {
			return nil, err
		}
		ex := Example{
			Board:  boards,
			Policy: policies,
			// THIS IS A HACK.
			// The value is 1 or -1 depending on player colour or draw 0,
			// but for now we store the player colour for this turn.
			Value: float32(a.game.Turn()),
		}
		if validPolicies(policies) {
			examples = append(examples, ex)
		}
		a.game = a.game.Apply(best)
	}

	for i := range examples {
		switch {
		case winner == chess.NoColor: // draw
			examples[i].Value = 0
		case examples[i].Value == float32(winner):
			examples[i].Value = 1
		default:
			examples[i].Value = -1
		}
	}

	a.CurrentAgent.MCTS.Reset()
	a.game.Reset()
	runtime.GC()

	a.CurrentAgent.MCTS = mcts.New(a.game, a.conf, a.CurrentAgent)
	if err := a.CurrentAgent.Close(); err != nil {
		return nil, err
	}

	return examples, nil
}

// Name of the game
func (a *Arena) Name() string { return a.name }

// State of the game
func (a *Arena) State() game.State { return a.game }

func validPolicies(policy []float32) bool {
	for _, v := range policy {
		if math32.IsInf(v, 0) {
			return false
		}
		if math32.IsNaN(v) {
			return false
		}
	}
	return true
}
