package game

import (
	"bufio"
	"log"
	"os"

	"github.com/notnil/chess"
)

type Chess struct {
	history     []*chess.Game
	actionSpace map[int64]string
}

// ChessGame returns new Chess game state.
// fileMoves is a file containing 'almost' all possible UCI notation moves
// each move is one line.
func ChessGame(movesFile string) *Chess {
	f, err := os.Open(movesFile)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)

	actionSpace := make(map[int64]string)
	var idx int64
	for scanner.Scan() {
		actionSpace[idx] = scanner.Text()
		idx++
	}

	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}

	g := chess.NewGame()
	return &Chess{
		history:     []*chess.Game{g},
		actionSpace: actionSpace,
	}
}

func (g *Chess) ActionSpace() int {
	return len(g.actionSpace)
}

