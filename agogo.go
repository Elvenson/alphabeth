package agogo

import (
	"encoding/gob"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	dual "github.com/alphabeth/dualnet"
	"github.com/alphabeth/game"
	"github.com/alphabeth/mcts"
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

// constant variables.
const (
	metaFile  = "meta.json"
	modelFile = "checkpoint.model"
)

// MetaData consists of exported params for model.
type MetaData struct {
	NNConf   dual.Config `json:"nn_conf"`
	MCTSConf mcts.Config `json:"mcts_conf"`
}

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

	if err := a.Init(); err != nil {
		panic(fmt.Sprintf("%+v", err))
	}

	retVal := &AZ{
		Arena:           MakeArena(g, a, conf.MCTSConf, conf.Encoder, conf.Name),
		nnConf:          conf.NNConf,
		mctsConf:        conf.MCTSConf,
		enc:             conf.Encoder,
		updateThreshold: float32(conf.UpdateThreshold),
		maxExamples:     conf.MaxExamples,
	}
	return retVal
}

// LearnAZ learns for iterations. It self-plays for episodes, and then trains a new NN from the self play example.
// The difference between this and `Learn` function is that in Alpha Zero we just simply store the latest model
// no need compete with the current best agent.
func (a *AZ) LearnAZ(iters, episodes, nniters int) error {
	var err error
	var exs []Example
	for epoch := 0; epoch < iters; epoch++ {
		log.Printf("Epoch %v\n", epoch)
		var examples []Example
		for e := 0; e < episodes; e++ {
			log.Printf("Episode %v\n", e)

			// generates training examples
			if exs, err = a.SelfPlay(); err != nil {
				return err
			}
			examples = append(examples, exs...)
		}

		if a.maxExamples > 0 && len(examples) > a.maxExamples {
			shuffleExamples(examples)
			examples = examples[:a.maxExamples]
		}
		Xs, Policies, Values, batches := a.prepareExamples(examples)

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

// SaveAZ saves AlphaZero into filename.
func (a *AZ) SaveAZ(dirName string) error {
	err := os.Mkdir(dirName, 0755)
	if err != nil {
		return err
	}

	// Save config.
	metaPath := filepath.Join(dirName, metaFile)
	metaConf := &MetaData{
		NNConf:   a.nnConf,
		MCTSConf: a.mctsConf,
	}
	jsonStr, err := json.MarshalIndent(metaConf, "", "	")
	if err != nil {
		return err
	}
	err = ioutil.WriteFile(metaPath, jsonStr, 0544)
	if err != nil {
		return err
	}

	modelPath := filepath.Join(dirName, modelFile)
	f, err := os.OpenFile(modelPath, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0544)
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

	a.CurrentAgent.NN = dual.New(a.nnConf)

	dec := gob.NewDecoder(f)
	if err = dec.Decode(a.CurrentAgent.NN); err != nil {
		return errors.WithStack(err)
	}
	return nil
}

// Load loads model based on checkpoint and meta data.
func Load(dirName string, g game.State, encoder func(g game.State) []float32) (*AZ, error) {
	metaPath := filepath.Join(dirName, metaFile)
	metaStr, err := ioutil.ReadFile(metaPath)
	if err != nil {
		return nil, err
	}
	metaConf := &MetaData{}
	err = json.Unmarshal(metaStr, metaConf)
	if err != nil {
		return nil, err
	}

	conf := Config{
		Name:     "Alphabeth",
		NNConf:   metaConf.NNConf,
		MCTSConf: metaConf.MCTSConf,
	}
	conf.Encoder = encoder

	modelPath := filepath.Join(dirName, modelFile)
	a := New(g, conf)
	err = a.Load(modelPath)
	if err != nil {
		return nil, err
	}

	return a, nil
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
