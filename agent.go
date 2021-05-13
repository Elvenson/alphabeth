package agogo

import (
	"log"
	"sync"

	"github.com/hashicorp/go-multierror"
	"github.com/notnil/chess"

	dual "github.com/alphabeth/dualnet"
	"github.com/alphabeth/game"
	"github.com/alphabeth/mcts"
)

// An Agent is a player, AI or Human
type Agent struct {
	NN     *dual.Dual
	MCTS   *mcts.MCTS
	Player chess.Color
	Enc    GameEncoder
	sync.Mutex
	name     string
	actions  int
	inferer  chan Inferer
	err      error
	inferers []Inferer
}

// SwitchToInference uses the inference mode neural network.
func (a *Agent) SwitchToInference() (err error) {
	a.Lock()
	a.inferer = make(chan Inferer, a.MCTS.NumSimulation)

	for i := 0; i < a.MCTS.NumSimulation; i++ {
		var inf Inferer
		if inf, err = dual.Infer(a.NN, false); err != nil {
			return err
		}
		a.inferers = append(a.inferers, inf)
		a.inferer <- inf
	}
	// a.NN = nil // remove old NN
	a.Unlock()
	return nil
}

// Infer infers a bunch of moves based on the game state.
// This is mainly used to implement a Inferer such that the MCTS search can use it.
func (a *Agent) Infer(g game.State) (policy []float32, value float32) {
	input := a.Enc(g)
	inf := <-a.inferer

	var err error
	policy, value, err = inf.Infer(input)
	if err != nil {
		if el, ok := inf.(ExecLogger); ok {
			log.Println(el.ExecLog())
		}
		panic(err)
	}
	a.inferer <- inf
	return
}

// Search searches the game state and returns a suggested move.
func (a *Agent) Search(g game.State) (game.Move, error) {
	a.MCTS.SetGame(g)
	return a.MCTS.Search()
}

// Close closes channel to free up memory.
func (a *Agent) Close() error {
	close(a.inferer)
	var errs error
	for _, inferer := range a.inferers {
		if err := inferer.Close(); err != nil {
			errs = multierror.Append(errs, err)
		}
	}
	if errs != nil {
		return errs
	}
	return nil
}
