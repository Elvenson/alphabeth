package agogo

import (
	"bytes"
	"fmt"
	"io"
	"log"
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
	bestAgent, currentAgent *Agent

	// state
	currentPlayer *Agent
	conf          mcts.Config
	buf           bytes.Buffer
	logger        *log.Logger

	// only relevant to training
	name       string
	epoch      int // training epoch
	gameNumber int // which game is this in

	// when to screw it all and just reinit a new NN
	oldThresh int
	oldCount  int
}

// MakeArena makes an arena given a game.
func MakeArena(g game.State, a, b Dualer, conf mcts.Config, enc GameEncoder, name string) Arena {
	bestAgent := &Agent{
		NN:   a.Dual(),
		Enc:  enc,
		name: "A",
	}
	bestAgent.MCTS = mcts.New(g, conf, bestAgent)
	currentAgent := &Agent{
		NN:   b.Dual(),
		Enc:  enc,
		name: "B",
	}
	currentAgent.MCTS = mcts.New(g, conf, currentAgent)

	if name == "" {
		name = "UNKNOWN GAME"
	}

	return Arena{
		r:            rand.New(rand.NewSource(time.Now().UnixNano())),
		game:         g,
		bestAgent:    bestAgent,
		currentAgent: currentAgent,
		conf:         conf,
		name:         name,

		oldThresh: 10,
	}
}

// NewArena makes an arena an returns a pointer to the Arena
func NewArena(g game.State, a, b Dualer, conf mcts.Config, enc GameEncoder, aug Augmenter, name string) *Arena {
	ar := MakeArena(g, a, b, conf, enc, name)
	ar.logger = log.New(&ar.buf, "", log.Ltime)
	return &ar
}

// Play plays a game, and returns a winner. If it is a draw, the returned colour is None.
func (a *Arena) Play(record bool) (examples []Example) {
	// set the current agent as white piece
	a.currentPlayer = a.currentAgent
	a.currentAgent.Player = chess.White
	a.bestAgent.Player = chess.Black
	a.logger.Printf("Playing. Recording %t\n", record)
	a.logger.SetPrefix("\t\t")

	var winner chess.Color
	var ended bool
	for ended, winner = a.game.Ended(); !ended; ended, winner = a.game.Ended() {
		best := a.currentPlayer.Search(a.game)
		if best == game.ResignMove {
			break
		}

		a.logger.Printf("Current Player: %v. Best Move %v\n", a.currentPlayer.Player, best)
		if record {
			boards := a.currentPlayer.Enc(a.game)
			policies := a.currentPlayer.MCTS.Policies(a.game)
			ex := Example{
				Board:  boards,
				Policy: policies,
				// THIS IS A HACK.
				// The value is 1 or -1 depending on player colour, but for now we store the player colour for this turn
				Value: float32(a.currentPlayer.Player),
			}
			if validPolicies(policies) {
				examples = append(examples, ex)
			}

		}

		a.game = a.game.Apply(best)
		a.switchPlayer()
	}
	a.logger.SetPrefix("\t")
	a.currentAgent.MCTS.Reset()
	a.bestAgent.MCTS.Reset()

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
	var winningAgent *Agent
	switch {
	case winner == chess.NoColor:
		a.currentAgent.Draw++
		a.bestAgent.Draw++
	case winner == a.currentAgent.Player:
		a.currentAgent.Wins++
		a.bestAgent.Loss++
		winningAgent = a.currentAgent
	case winner == a.bestAgent.Player:
		a.bestAgent.Wins++
		a.currentAgent.Loss++
		winningAgent = a.bestAgent
	}
	if !record {
		log.Printf("Winner %v | %p", winner, winningAgent)
	}
	a.currentAgent.MCTS = mcts.New(a.game, a.conf, a.currentAgent)
	a.bestAgent.MCTS = mcts.New(a.game, a.conf, a.bestAgent)
	runtime.GC()
	return examples
}

// Epoch returns the current Epoch
func (a *Arena) Epoch() int { return a.epoch }

// GameNumber returns the
func (a *Arena) GameNumber() int { return a.gameNumber }

// Name of the game
func (a *Arena) Name() string { return a.name }

// State of the game
func (a *Arena) State() game.State { return a.game }

// Log the MCTS of both players into w
func (a *Arena) Log(w io.Writer) {
	fmt.Fprintf(w, a.buf.String())
	fmt.Fprintf(w, "\nA:\n\n")
	fmt.Fprintln(w, a.currentAgent.MCTS.Log())
	fmt.Fprintf(w, "\nB:\n\n")
	fmt.Fprintln(w, a.bestAgent.MCTS.Log())
}

func (a *Arena) newAgent(conf dual.Config, killedA bool) (err error) {
	if killedA || a.oldCount >= a.oldThresh {
		a.currentAgent.NN = dual.New(conf)
		err = a.currentAgent.NN.Init()
		if err != nil {
			return err
		}
		a.oldCount = 0
	}
	a.oldCount++
	log.Printf("NewB NN %p", a.currentAgent.NN)
	return err
}

func (a *Arena) switchPlayer() {
	switch a.currentPlayer {
	case a.currentAgent:
		a.currentPlayer = a.bestAgent
	case a.bestAgent:
		a.currentPlayer = a.currentAgent
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
