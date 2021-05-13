package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"

	agogo "github.com/alphabeth"
	"github.com/alphabeth/game"
	"github.com/notnil/chess"
)

var (
	fileMoves = flag.String("moves_file", "", "file containing chess moves")
	dirName   = flag.String("model_path", "", "directory contains trained model")
)

func main() {
	flag.Parse()
	g := game.ChessGame(*fileMoves)
	az, err := agogo.Load(*dirName, g, game.InputEncoder)
	if err != nil {
		fmt.Printf("error loading model: %s\n", err)
	}
	var winner chess.Color
	var ended bool
	err = az.CurrentAgent.SwitchToInference()
	if err != nil {
		panic(err)
	}
	for ended, winner = g.Ended(); !ended; ended, winner = g.Ended() {
		best, err := az.CurrentAgent.Search(g)
		if err !=nil {
			panic(err)
		}
		if best == game.ResignMove {
			break
		}
		g = g.Apply(best).(*game.Chess)
		g.ShowBoard()
		fmt.Printf("possible moves is: %+v\n", g.Moves())
		fmt.Printf("waiting for user move\n")
		input := bufio.NewScanner(os.Stdin)
		input.Scan()
		g = g.Apply(game.Move(input.Text())).(*game.Chess)
		g.ShowBoard()
	}
	fmt.Printf("winner is %s", winner.String())
}
