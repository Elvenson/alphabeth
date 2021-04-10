package agogo

import (
	"encoding/gob"
	"fmt"
	"log"
	"math/rand"
	"os"
	"time"

	dual "github.com/alphabeth/dualnet"
	"github.com/alphabeth/game"
	"github.com/alphabeth/mcts"
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

// AZ is the top level structure and the entry point of the API.
// It it a wrapper around the MTCS and the Neural Network that composes the algorithm.
// AZ stands for AlphaZero
type AZ struct {
	// state
	Arena

	// config
	nnConf          dual.Config
	mctsConf        mcts.Config
	enc             GameEncoder
	updateThreshold float32
	maxExamples     int
}

// New AlphaZero structure. It takes a game state (implementing the board, rules, etc.)
// and a configuration to apply to the MCTS and the neural network
func New(g game.State, conf Config) *AZ {
	if !conf.NNConf.IsValid() {
		panic("NNConf is not valid. Unable to proceed")
	}
	if !conf.MCTSConf.IsValid() {
		panic("MCTSConf is not valid. Unable to proceed")
	}

	a := dual.New(conf.NNConf)
	b := dual.New(conf.NNConf)

	if err := a.Init(); err != nil {
		panic(fmt.Sprintf("%+v", err))
	}
	if err := b.Init(); err != nil {
		panic(fmt.Sprintf("%+v", err))
	}

	retVal := &AZ{
		Arena:           MakeArena(g, a, b, conf.MCTSConf, conf.Encoder, conf.Name),
		nnConf:          conf.NNConf,
		mctsConf:        conf.MCTSConf,
		enc:             conf.Encoder,
		updateThreshold: float32(conf.UpdateThreshold),
		maxExamples:     conf.MaxExamples,
	}
	retVal.logger = log.New(&retVal.buf, "", log.Ltime)
	return retVal
}

// Learn learns for iterations. It self-plays for episodes, and then trains a new NN from the self play example.
func (a *AZ) Learn(iters, episodes, nniters, arenaGames int) error {
	var err error
	for a.epoch = 0; a.epoch < iters; a.epoch++ {
		var ex []Example
		log.Printf("Self Play for epoch %d. Current best agent %p, current agent %p",
			a.epoch, a.bestAgent, a.currentAgent)

		a.buf.Reset()
		a.logger.SetPrefix("\t")

		for e := 0; e < episodes; e++ {
			log.Printf("Episode %v\n", e)

			// generates training examples
			if exs, err := a.SelfPlay(); err != nil {
				return err
			} else {
				ex = append(ex, exs...)
			}
		}
		a.logger.SetPrefix("")
		a.buf.Reset()

		if a.maxExamples > 0 && len(ex) > a.maxExamples {
			shuffleExamples(ex)
			ex = ex[:a.maxExamples]
		}
		Xs, Policies, Values, batches := a.prepareExamples(ex)

		if err = dual.Train(a.currentAgent.NN, Xs, Policies, Values, batches, nniters); err != nil {
			return errors.WithMessage(err, fmt.Sprintf("Train fail"))
		}

		if err = a.currentAgent.SwitchToInference(); err != nil {
			return err
		}
		if err = a.bestAgent.SwitchToInference(); err != nil {
			return err
		}

		a.currentAgent.resetStats()
		a.bestAgent.resetStats()

		a.logger.Printf("Playing Arena")
		a.logger.SetPrefix("\t")
		for a.gameNumber = 0; a.gameNumber < arenaGames; a.gameNumber++ {
			a.logger.Printf("Playing game number %d", a.gameNumber)
			a.Play()
		}
		a.logger.SetPrefix("")

		if err = a.bestAgent.Close(); err != nil {
			return err
		}
		if err = a.currentAgent.Close(); err != nil {
			return err
		}

		var killedA bool
		log.Printf("Current best agent wins %v, loss %v, draw %v\ncurrent agent wins %v, loss %v, draw %v",
			a.bestAgent.Wins, a.bestAgent.Loss, a.bestAgent.Draw,
			a.currentAgent.Wins, a.currentAgent.Loss, a.currentAgent.Draw)

		// if a.B.Wins/(a.B.Wins+a.B.Loss+a.B.Draw) > a.updateThreshold {
		if a.currentAgent.Wins/(a.currentAgent.Wins+a.bestAgent.Wins) > a.updateThreshold {
			// B wins. Kill A, clean up its resources.
			log.Printf("Kill current best agent %p. New best agent's NN is %p", a.bestAgent.NN, a.currentAgent.NN)
			a.bestAgent.NN = a.currentAgent.NN
			// clear examples
			ex = ex[:0]
			killedA = true
		}

		// creat new agent if met condition
		if err = a.newAgent(a.nnConf, killedA); err != nil {
			return err
		}
	}
	return nil
}

// Save learning into filename.
func (a *AZ) Save(filename string) error {
	f, err := os.OpenFile(filename, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0544)
	if err != nil {
		return err
	}
	defer f.Close()

	enc := gob.NewEncoder(f)
	return enc.Encode(a.bestAgent.NN)
}

// Load the Alpha Zero structure from a filename
func (a *AZ) Load(filename string) error {
	f, err := os.Open(filename)
	if err != nil {
		return errors.WithStack(err)
	}
	defer f.Close()

	a.bestAgent.NN = dual.New(a.nnConf)
	a.currentAgent.NN = dual.New(a.nnConf)

	dec := gob.NewDecoder(f)
	if err = dec.Decode(a.bestAgent.NN); err != nil {
		return errors.WithStack(err)
	}

	f.Seek(0, 0)
	dec = gob.NewDecoder(f)
	if err = dec.Decode(a.currentAgent.NN); err != nil {
		return errors.WithStack(err)
	}
	return nil
}

func (a *AZ) prepareExamples(examples []Example) (Xs, Policies, Values *tensor.Dense, batches int) {
	shuffleExamples(examples)
	batches = len(examples) / a.nnConf.BatchSize
	total := batches * a.nnConf.BatchSize
	var XsBacking, PoliciesBacking, ValuesBacking []float32
	for i, ex := range examples {
		if i >= total {
			break
		}
		XsBacking = append(XsBacking, ex.Board...)

		start := len(PoliciesBacking)
		PoliciesBacking = append(PoliciesBacking, make([]float32, len(ex.Policy))...)
		copy(PoliciesBacking[start:], ex.Policy)

		ValuesBacking = append(ValuesBacking, ex.Value)
	}

	actionSpace := a.Arena.game.ActionSpace()
	Xs = tensor.New(tensor.WithBacking(XsBacking), tensor.WithShape(a.nnConf.BatchSize*batches, a.nnConf.Features, a.nnConf.Height, a.nnConf.Width))
	Policies = tensor.New(tensor.WithBacking(PoliciesBacking), tensor.WithShape(a.nnConf.BatchSize*batches, actionSpace))
	Values = tensor.New(tensor.WithBacking(ValuesBacking), tensor.WithShape(a.nnConf.BatchSize*batches))
	return
}

func shuffleExamples(examples []Example) {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	for i := range examples {
		j := r.Intn(i + 1)
		examples[i], examples[j] = examples[j], examples[i]
	}
}
