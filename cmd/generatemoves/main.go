// This package is for generating nearly all possible moves and write them into file for model to read and
// encodes each of them as one hot encoding.

package main

import (
	"flag"
	"log"
	"math/rand"
	"os"

	"github.com/notnil/chess"
)

var (
	numGameFlag   = flag.Int("num_game", 10, "number of game to play")
	chessMovePath = flag.String("path", "chess_moves.txt", "chess possible moves path to generate to")
)

func main() {
	flag.Parse()

	// If the file doesn't exist, create it, or append to the file
	f, err := os.OpenFile(*chessMovePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	movesMap := make(map[string]struct{}, 0)
	for i := 0; i < *numGameFlag; i++ {
		game := chess.NewGame()
		// generate moves until game is over
		for game.Outcome() == chess.NoOutcome {
			// select a random move
			moves := game.ValidMoves()
			for _, m := range moves {
				mStr := m.String()
				if _, ok := movesMap[mStr]; !ok {
					movesMap[mStr] = struct{}{}
					if _, err := f.Write([]byte(mStr + "\n")); err != nil {
						log.Fatal(err)
					}
				}
			}
			move := moves[rand.Intn(len(moves))]
			if err := game.Move(move); err != nil {
				log.Fatal(err)
			}
		}
	}

}
