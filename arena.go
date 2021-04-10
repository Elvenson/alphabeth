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

func setupSelfPlay(agent *Agent) error {
	if err := agent.SwitchToInference(); err != nil {
		return err
	}
	return nil
}

// SelfPlays lets the agent to generate training data by playing with itself.
func (a *Arena) SelfPlay() (examples []Example, err error) {
	a.logger.Printf("Self playing\n")
	a.logger.SetPrefix("\t\t")
	if err := setupSelfPlay(a.currentAgent); err != nil {
		return nil, err
	}

	var winner chess.Color
	var ended bool
	for ended, winner = a.game.Ended(); !ended; ended, winner = a.game.Ended() {
		best := a.currentAgent.Search(a.game)
		if best == game.ResignMove {
			break
		}

		log.Printf("Current Player: %v. Best Move %v\n", a.game.Turn(), best)
		boards := a.currentAgent.Enc(a.game)
		policies := a.currentAgent.MCTS.Policies(a.game)
		ex := Example{
			Board:  boards,
			Policy: policies,
			// THIS IS A HACK.
			// The value is 1 or -1 depending on player colour, but for now we store the player colour for this turn
			Value: float32(a.currentAgent.Player),
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

	a.currentAgent.MCTS.Reset()
	a.game.Reset()
	runtime.GC()

	a.currentAgent.MCTS = mcts.New(a.game, a.conf, a.currentAgent)

	return examples, nil
}

// Play plays a game, and records who is the winner. If it is a draw, the returned colour is None.
func (a *Arena) Play() {
	// set the current agent as white piece
	a.currentPlayer = a.currentAgent
	a.currentAgent.Player = chess.White
	a.bestAgent.Player = chess.Black

	var winner chess.Color
	var ended bool
	for ended, winner = a.game.Ended(); !ended; ended, winner = a.game.Ended() {
		best := a.currentPlayer.Search(a.game)
		if best == game.ResignMove {
			break
		}
		a.game = a.game.Apply(best)
		a.switchPlayer()
	}

	a.currentAgent.MCTS.Reset()
	a.bestAgent.MCTS.Reset()
	a.game.Reset()
	runtime.GC()

	switch {
	case winner == chess.NoColor:
		a.currentAgent.Draw++
		a.bestAgent.Draw++
	case winner == a.currentAgent.Player:
		a.currentAgent.Wins++
		a.bestAgent.Loss++
	case winner == a.bestAgent.Player:
		a.bestAgent.Wins++
		a.currentAgent.Loss++
	}

	a.currentAgent.MCTS = mcts.New(a.game, a.conf, a.currentAgent)
	a.bestAgent.MCTS = mcts.New(a.game, a.conf, a.bestAgent)

	return
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
		log.Printf("NewB NN %p", a.currentAgent.NN)
		a.currentAgent.NN = dual.New(conf)
		err = a.currentAgent.NN.Init()
		if err != nil {
			return err
		}
		a.oldCount = 0
	}
	a.oldCount++
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
