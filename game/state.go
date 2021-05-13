package game

import "github.com/notnil/chess"

// Move encodes chess move with UCI notation
type Move string

const (
	Begin      = -3
	Resign     = -2
	ResignMove = Move("resign")
	RowNum     = 8
	ColNum     = 8
)

// State is any game that implements these and are able to report back
type State interface {
	// These methods represent the game state
	ActionSpace() int                 // returns the number of permissible actions
	Board() *chess.Board              // return board state.
	Turn() chess.Color                // Turn returns the color to move next.
	MoveNumber() int                  // returns count of moves so far that led to this point.
	LastMove() int32                  // returns the last move that was made in neural network index.
	NNToMove(idx int32) (Move, error) // returns move from neural network encoding output space.

	// Meta-game stuff
	Ended() (ended bool, winner chess.Color) // has the game ended? if yes, then who's the winner?
	Resign(color chess.Color)                // current player resign the game.

	// interactions
	Check(m Move) bool      // check if the placement is legal.
	Apply(m Move) State     // should return a GameState. The required side effect is the NextToMove has to change.
	Reset()                 // reset state.
	PossibleMoves() []int32 // get all possible index moves.

	// For MCTS
	UndoLastMove()
	Fwd()

	// generics
	Eq(other State) bool
	Clone() State
	ShowBoard()
}
