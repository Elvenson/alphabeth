package game

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"sync"

	"github.com/notnil/chess"
)

// Chess struct
type Chess struct {
	sync.Mutex
	history            []chess.Game
	actionSpace        map[int32]Move
	reverseActionSpace map[Move]int32
	histPtr            int
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

	actionSpace := make(map[int32]Move)
	reverseActionSpace := make(map[Move]int32)
	var idx int32
	for scanner.Scan() {
		m := Move(scanner.Text())
		actionSpace[idx] = m
		reverseActionSpace[m] = idx
		idx++
	}

	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}

	// new game with UCI notation
	g := chess.NewGame(chess.UseNotation(chess.UCINotation{}))
	return &Chess{
		Mutex:              sync.Mutex{},
		history:            []chess.Game{*g},
		actionSpace:        actionSpace,
		reverseActionSpace: reverseActionSpace,
		histPtr:            0,
	}
}

// ActionSpace returns the number of permissible actions.
func (g *Chess) ActionSpace() int {
	return len(g.actionSpace)
}

// Board returns board state.
func (g *Chess) Board() *chess.Board {
	return g.history[g.histPtr].Position().Board()
}

// Turn returns the color to move next.
func (g *Chess) Turn() chess.Color {
	return g.history[g.histPtr].Position().Turn()
}

// MoveNumber returns count of moves so far that led to this point.
func (g *Chess) MoveNumber() int {
	return g.histPtr
}

// LastMove returns the last move that was made in neural network index.
func (g *Chess) LastMove() int32 {
	var idx int32
	var ok bool
	if g.histPtr == 0 { // at the beginning no move
		return -1
	}
	lastG := g.history[g.histPtr]
	moveHist := lastG.Moves()
	m := Move(moveHist[len(moveHist)-1].String())
	if idx, ok = g.reverseActionSpace[m]; !ok {
		log.Panicf("move out of range: %s", m)
	}
	return idx
}

// NNToMove returns move from neural network encoding output space.
func (g *Chess) NNToMove(idx int32) (Move, error) {
	var m Move
	var ok bool
	if m, ok = g.actionSpace[idx]; !ok {
		return "", fmt.Errorf("invalid index: %d", idx)
	}
	return m, nil
}

// Ended returns true if ended and the winner color.
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

// Resign resigns the game and mark the game as ended.
func (g *Chess) Resign(color chess.Color) {
	g.history[g.histPtr].Resign(color)
}

// Check checks if the placement is legal.
func (g *Chess) Check(m Move) bool {
	moves := g.history[g.histPtr].ValidMoves()
	for _, move := range moves {
		a := move.String()
		if m == Move(a) {
			return true
		}
	}
	return false
}

// Apply applies move and return new state.
func (g *Chess) Apply(m Move) State {
	newG := g.history[g.histPtr].Clone()
	err := newG.MoveStr(string(m))
	if err != nil {
		panic(err)
	}
	g.histPtr++
	if g.histPtr > len(g.history) {
		panic(fmt.Sprintf("history pointer %d cannot be larger than history len %d",
			g.histPtr, len(g.history)))
	}
	if g.histPtr == len(g.history) {
		g.history = append(g.history, *newG)
	} else {
		g.history[g.histPtr] = *newG
	}
	return g
}

// PossibleMoves gets all possible moves in output index format.
func (g *Chess) PossibleMoves() []int32 {
	moves := g.history[g.histPtr].ValidMoves()
	mIdx := make([]int32, len(moves))
	for i, m := range moves {
		mIdx[i] = g.reverseActionSpace[Move(m.String())]
	}
	return mIdx
}

// Reset resets state.
func (g *Chess) Reset() {
	g.history = g.history[:1] // reset to first state
	g.histPtr = 0
}

// UndoLastMove undoes last move.
func (g *Chess) UndoLastMove() {
	if g.histPtr > 0 {
		g.histPtr--
	}
}

// Fwd forwards move history
func (g *Chess) Fwd() {
	if g.histPtr < len(g.history)-1 {
		g.histPtr++
	}
}

// Eq checks if 2 stats are equal or not.
func (g *Chess) Eq(other State) bool {
	ot, ok := other.(*Chess)
	if !ok {
		panic("cannot cast to chess game state")
	}
	return ot.history[ot.histPtr].Position().Hash() == g.history[g.histPtr].Position().Hash()

}

// Clone clones state.
func (g *Chess) Clone() State {
	g.Lock()
	n := &Chess{
		Mutex:              sync.Mutex{},
		history:            make([]chess.Game, len(g.history)),
		actionSpace:        make(map[int32]Move, 0),
		reverseActionSpace: make(map[Move]int32, 0),
		histPtr:            g.histPtr,
	}
	copy(n.history, g.history)
	for k, v := range g.actionSpace {
		n.actionSpace[k] = v
	}
	for k, v := range g.reverseActionSpace {
		n.reverseActionSpace[k] = v
	}

	g.Unlock()
	return n
}

// ShowBoard show the current board position.
func (g *Chess) ShowBoard() {
	fmt.Println(g.history[g.histPtr].Position().Board().Draw())
}

func (g *Chess) Moves() []Move {
	moves := g.history[g.histPtr].ValidMoves()
	moveList := make([]Move, len(moves))
	for i, m := range moves {
		moveList[i] = Move(m.String())
	}
	return moveList
}
