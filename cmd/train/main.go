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
		MCTSConf:        mcts.DefaultConfig(8),
		UpdateThreshold: 0.52,
	}

	conf.NNConf.BatchSize = 100
	conf.NNConf.Features = 2 // write a better encoding of the board, and increase features (and that allows you to increase K as well)
	conf.NNConf.K = 3
	conf.NNConf.SharedLayers = 3
	conf.MCTSConf = mcts.Config{
		PUCT:        1.0,
		M:           3,
		N:           3,
		Timeout:     100 * time.Millisecond,
		Budget:      1000,
		RandomCount: 0,
	}

	conf.Encoder = game.InputEncoder

	a := agogo.New(g, conf)
	if err := a.Learn(5, 30, 200, 30); err != nil {
		log.Fatalf("error when learning chess: %s", err)
	}

	if err := a.Save("example.model"); err != nil {
		log.Fatalf("error when saving model: %s", err)
	}
}
