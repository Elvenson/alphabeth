package game

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"sync"

	"github.com/notnil/chess"
)

type Chess struct {
	sync.Mutex
	history     []*chess.Game
	actionSpace map[int64]Move
	histPtr     int
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

	actionSpace := make(map[int64]Move)
	var idx int64
	for scanner.Scan() {
		actionSpace[idx] = Move(scanner.Text())
		idx++
	}

	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}

	// new game with UCI notation.
	g := chess.NewGame()
	return &Chess{
		Mutex:       sync.Mutex{},
		history:     []*chess.Game{g},
		actionSpace: actionSpace,
		histPtr:     0,
	}
}

func (g *Chess) ActionSpace() int {
	return len(g.actionSpace)
}

func (g *Chess) Hash() [16]byte {
	return g.history[g.histPtr].Position().Hash()
}

func (g *Chess) Turn() chess.Color {
	return g.history[g.histPtr].Position().Turn()
}

func (g *Chess) MoveNumber() int {
	return g.histPtr
}

func (g *Chess) LastMove() Move {
	if g.histPtr == 0 { // at the beginning no move.
		return ""
	}
	lastG := g.history[g.histPtr]
	moveHist := lastG.Moves()
	return Move(moveHist[len(moveHist)-1].String())
}

func (g *Chess) NNToMove(idx int64) Move {
	if m, ok := g.actionSpace[idx]; !ok {
		panic(fmt.Sprintf("index out of range: %d", idx))
	} else {
		return m
	}
}

func (g *Chess) Ended() (ended bool, winner chess.Color) {
	r := g.history[g.histPtr].Outcome()
	if r == chess.NoOutcome {
		ended = false
		winner = chess.NoColor
		return
	}
	ended = true

	if r == chess.Draw {
		winner = chess.NoColor
	} else if r == chess.BlackWon {
		winner = chess.Black
	} else {
		winner = chess.White
	}
	return
}

func (g *Chess) Score(p chess.Color) float32 {
	r := g.history[g.histPtr].Outcome()
	if r == chess.Draw || r == chess.NoOutcome {
		return 0
	}

	if r == chess.BlackWon {
		if p == chess.Black {
			return 1
		} else {
			return -1
		}
	} else {
		if p == chess.Black {
			return -1
		} else {
			return 1
		}
	}
}

func (g *Chess) Check(m Move) bool {
	moves := g.history[g.histPtr].ValidMoves()
	for _, move := range moves {
		if m == Move(move.String()) {
			return true
		}
	}
	return false
}

func (g *Chess) Apply(m Move) State {
	newG := g.history[g.histPtr].Clone()
	err := newG.MoveStr(string(m))
	if err != nil {
		panic(err)
	}
	g.histPtr++
	if g.histPtr >= len(g.history) {
		g.history = append(g.history, newG)
	} else {
		g.history[g.histPtr] = newG
	}
	return g
}

func (g *Chess) Reset() {
	g.history = g.history[:1] // reset to first state.
	g.histPtr = 0
}

func (g *Chess) UndoLastMove() {
	if g.histPtr > 0 {
		g.histPtr--
	}
}

func (g *Chess) Fwd() {
	if g.histPtr < len(g.history)-1 {
		g.histPtr++
	}
}

func (g *Chess) InputEncoder() []float32 {
	m := g.history[g.histPtr].Position().Board().SquareMap()
	board := make([]float32, len(m))
	for k, v := range m {
		if v == chess.NoPiece {
			board[int8(k)] = 0.001
		} else {
			board[int8(k)] = float32(v)
		}
	}

	playerLayer := make([]float32, len(m))
	next := g.Turn()
	for i := range playerLayer {
		playerLayer[i] = float32(next)
	}
	inputLayer := append(board, playerLayer...)
	return inputLayer
}

func (g *Chess) Eq(other State) bool {
	ot, ok := other.(*Chess)
	if !ok {
		panic("cannot cast to chess game state")
	}
	return ot.history[ot.histPtr].Position().Hash() == g.history[g.histPtr].Position().Hash()

}

func (g *Chess) Clone() State {
	g.Lock()
	n := &Chess{
		Mutex:       sync.Mutex{},
		history:     nil,
		actionSpace: g.actionSpace,
		histPtr:     g.histPtr,
	}
	copy(n.history, g.history)
	g.Unlock()
	return n
}
