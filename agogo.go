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
	return retVal
}

// Learn learns for iterations. It self-plays for episodes, and then trains a new NN from the self play example.
// Finally we let 2 agents compete then get the best agent based on threshold.
func (a *AZ) Learn(iters, episodes, nniters, arenaGames int) error {
	var err error
	for epoch := 0; epoch < iters; epoch++ {
		var ex []Example
		log.Printf("Self Play for epoch %d. Current best agent %p, current agent %p",
			epoch, a.BestAgent, a.CurrentAgent)

		for e := 0; e < episodes; e++ {
			log.Printf("Episode %v\n", e)

			// generates training examples
			if exs, err := a.SelfPlay(); err != nil {
				return err
			} else {
				ex = append(ex, exs...)
			}
		}

		if a.maxExamples > 0 && len(ex) > a.maxExamples {
			shuffleExamples(ex)
			ex = ex[:a.maxExamples]
		}
		Xs, Policies, Values, batches := a.prepareExamples(ex)

		if batches == 0 {
			return errors.New("batches is nil, probably too few examples regarding the batchsize")
		}

		log.Print("begin training")
		if err = dual.Train(a.CurrentAgent.NN, Xs, Policies, Values, batches, nniters); err != nil {
			return errors.WithMessage(err, fmt.Sprintf("Train fail"))
		}

		if err = a.CurrentAgent.SwitchToInference(a.game); err != nil {
			return err
		}
		if err = a.BestAgent.SwitchToInference(a.game); err != nil {
			return err
		}

		a.CurrentAgent.resetStats()
		a.BestAgent.resetStats()

		log.Print("Playing Arena")
		for a.gameNumber = 0; a.gameNumber < arenaGames; a.gameNumber++ {
			log.Printf("Playing game number %d", a.gameNumber)
			err := a.Play()
			if err != nil {
				return err
			}
		}

		if err = a.BestAgent.Close(); err != nil {
			return err
		}
		if err = a.CurrentAgent.Close(); err != nil {
			return err
		}

		var killedA bool
		log.Printf("Current best agent wins %v, loss %v, draw %v\ncurrent agent wins %v, loss %v, draw %v",
			a.BestAgent.Wins, a.BestAgent.Loss, a.BestAgent.Draw,
			a.CurrentAgent.Wins, a.CurrentAgent.Loss, a.CurrentAgent.Draw)

		// plus 1 in case all draw it will be divided by 0
		if a.CurrentAgent.Wins/(a.CurrentAgent.Wins+a.BestAgent.Wins + 1) > a.updateThreshold {
			// B wins. Kill A, clean up its resources
			log.Printf("Kill current best agent %p. New best agent's NN is %p", a.BestAgent.NN, a.CurrentAgent.NN)
			a.BestAgent.NN = a.CurrentAgent.NN
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

// Learn learns for iterations. It self-plays for episodes, and then trains a new NN from the self play example.
// The difference between this and `Learn` function is that in Alpha Zero we just simply store the latest model
// no need compete with the current best agent.
func (a *AZ) LearnAZ(iters, episodes, nniters int) error {
	var err error
	for epoch := 0; epoch < iters; epoch++ {
		var ex []Example
		for e := 0; e < episodes; e++ {
			log.Printf("Episode %v\n", e)

			// generates training examples
			if exs, err := a.SelfPlay(); err != nil {
				return err
			} else {
				ex = append(ex, exs...)
			}
		}

		if a.maxExamples > 0 && len(ex) > a.maxExamples {
			shuffleExamples(ex)
			ex = ex[:a.maxExamples]
		}
		Xs, Policies, Values, batches := a.prepareExamples(ex)

		if batches == 0 {
			return errors.New("batches is nil, probably too few examples regarding the batchsize")
		}

		log.Print("begin training")
		if err = dual.Train(a.CurrentAgent.NN, Xs, Policies, Values, batches, nniters); err != nil {
			return errors.WithMessage(err, fmt.Sprintf("Train fail"))
		}
	}
	return nil
}

// Save AlphaGo into filename.
func (a *AZ) Save(filename string) error {
	f, err := os.OpenFile(filename, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0544)
	if err != nil {
		return err
	}
	defer f.Close()

	enc := gob.NewEncoder(f)
	return enc.Encode(a.BestAgent.NN)
}

// Save AlphaZero into filename.
func (a *AZ) SaveAZ(filename string) error {
	f, err := os.OpenFile(filename, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0544)
	if err != nil {
		return err
	}
	defer f.Close()

	enc := gob.NewEncoder(f)
	return enc.Encode(a.CurrentAgent.NN)
}

// Load loads the Alpha model structure from a filename.
func (a *AZ) Load(filename string) error {
	f, err := os.Open(filename)
	if err != nil {
		return errors.WithStack(err)
	}
	defer f.Close()

	a.BestAgent.NN = dual.New(a.nnConf)
	a.CurrentAgent.NN = dual.New(a.nnConf)

	dec := gob.NewDecoder(f)
	if err = dec.Decode(a.BestAgent.NN); err != nil {
		return errors.WithStack(err)
	}

	f.Seek(0, 0)
	dec = gob.NewDecoder(f)
	if err = dec.Decode(a.CurrentAgent.NN); err != nil {
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
