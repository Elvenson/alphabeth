package main

import (
	"flag"
	"time"

	"log"

	agogo "github.com/alphabeth"
	dual "github.com/alphabeth/dualnet"
	"github.com/alphabeth/game"
	"github.com/alphabeth/mcts"
)

var (
	fileMoves = flag.String("moves_file", "", "file containing chess moves")
)

func main() {
	flag.Parse()

	g := game.ChessGame(*fileMoves)

	conf := agogo.Config{
		Name:            "Alphabeth",
		NNConf:          dual.DefaultConf(game.RowNum, game.ColNum, g.ActionSpace()),
		MCTSConf:        mcts.DefaultConfig(),
		UpdateThreshold: 0.55,
	}

	conf.NNConf.BatchSize = 20
	conf.NNConf.Features = 2 // write a better encoding of the board, and increase features (and that allows you to increase K as well)
	conf.NNConf.K = 3
	conf.NNConf.SharedLayers = 3
	conf.MCTSConf = mcts.Config{
		PUCT:        1.0,
		Timeout:     10000 * time.Millisecond,
		Budget:      1000,
		RandomCount: 10,
		MaxDepth:    10000,
		NumSimulation: 800,
		RandomMinVisits: 0,
		RandomTemperature: 10,
	}

	conf.Encoder = game.InputEncoder

	a := agogo.New(g, conf)
	if err := a.LearnAZ(1, 5, 5); err != nil {
		log.Fatalf("error when learning chess: %s", err)
	}

	log.Printf("Save model")
	if err := a.SaveAZ("example.model"); err != nil {
		log.Fatalf("error when saving model: %s", err)
	}
}
