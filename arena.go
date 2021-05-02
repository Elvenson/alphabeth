package agogo

import (
	"math/rand"
	"runtime"
	"time"

	dual "github.com/alphabeth/dualnet"
	"github.com/alphabeth/game"
	"github.com/alphabeth/mcts"
	"github.com/chewxy/math32"
	"github.com/notnil/chess"
)

// Arena represents a game arena
// Arena fulfils the interface game.MetaState
type Arena struct {
	r                       *rand.Rand
	game                    game.State
	BestAgent, CurrentAgent *Agent

	// state
	currentPlayer *Agent
	conf          mcts.Config

	// only relevant to training
	name       string
	gameNumber int // which game is this in

	// when to screw it all and just reinit a new NN
	oldThresh int
	oldCount  int
}

// MakeArena makes an arena given a game.
func MakeArena(g game.State, a, b Dualer, conf mcts.Config, enc GameEncoder, name string) Arena {
	BestAgent := &Agent{
		NN:   a.Dual(),
		Enc:  enc,
		name: "best agent",
	}
	BestAgent.MCTS = mcts.New(g, conf, BestAgent)
	CurrentAgent := &Agent{
		NN:   b.Dual(),
		Enc:  enc,
		name: "current agent",
	}
	CurrentAgent.MCTS = mcts.New(g, conf, CurrentAgent)

	return Arena{
		r:            rand.New(rand.NewSource(time.Now().UnixNano())),
		game:         g,
		BestAgent:    BestAgent,
		CurrentAgent: CurrentAgent,
		conf:         conf,
		name:         name,
		oldThresh:    10,
	}
}

// SelfPlays lets the agent to generate training data by playing with itself.
func (a *Arena) SelfPlay() (examples []Example, err error) {
	if err := a.CurrentAgent.SwitchToInference(a.game); err != nil {
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
		policies, err := a.CurrentAgent.MCTS.Policies(a.game)
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
	//if err := a.CurrentAgent.Close(); err != nil {
	//	return nil, err
	//}

	return examples, nil
}

// Play plays a game, and records who is the winner
func (a *Arena) Play() error {
	// set the current agent as white piece
	a.currentPlayer = a.CurrentAgent
	a.CurrentAgent.Player = chess.White
	a.BestAgent.Player = chess.Black

	var winner chess.Color
	var ended bool
	for ended, winner = a.game.Ended(); !ended; ended, winner = a.game.Ended() {
		best, err := a.currentPlayer.Search(a.game)
		if err != nil {
			return err
		}
		if best == game.ResignMove {
			break
		}
		a.game = a.game.Apply(best)
		a.switchPlayer()
	}

	a.CurrentAgent.MCTS.Reset()
	a.BestAgent.MCTS.Reset()
	a.game.Reset()
	runtime.GC()

	switch {
	case winner == chess.NoColor:
		a.CurrentAgent.Draw++
		a.BestAgent.Draw++
	case winner == a.CurrentAgent.Player:
		a.CurrentAgent.Wins++
		a.BestAgent.Loss++
	case winner == a.BestAgent.Player:
		a.BestAgent.Wins++
		a.CurrentAgent.Loss++
	}

	a.CurrentAgent.MCTS = mcts.New(a.game, a.conf, a.CurrentAgent)
	a.BestAgent.MCTS = mcts.New(a.game, a.conf, a.BestAgent)

	return nil
}

// GameNumber returns the
func (a *Arena) GameNumber() int { return a.gameNumber }

// Name of the game
func (a *Arena) Name() string { return a.name }

// State of the game
func (a *Arena) State() game.State { return a.game }

func (a *Arena) newAgent(conf dual.Config, killedA bool) (err error) {
	if killedA || a.oldCount >= a.oldThresh {
		a.CurrentAgent.NN = dual.New(conf)
		err = a.CurrentAgent.NN.Init()
		if err != nil {
			return err
		}
		a.oldCount = 0
	} else {
		a.oldCount++
	}
	return err
}

func (a *Arena) switchPlayer() {
	switch a.currentPlayer {
	case a.CurrentAgent:
		a.currentPlayer = a.BestAgent
	case a.BestAgent:
		a.currentPlayer = a.CurrentAgent
	}
}

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
