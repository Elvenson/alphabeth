package game

import "github.com/notnil/chess"

// Move encodes chess move with UCI notation
type Move string

// State is any game that implements these and are able to report back
type State interface {
	// These methods represent the game state
	ActionSpace() int        // returns the number of permissible actions
	Hash() [16]byte          // returns the hash of the board
	Turn() chess.Color       // Turn returns the color to move next.
	MoveNumber() int         // returns count of moves so far that led to this point.
	LastMove() Move          // returns the last move that was made
	NNToMove(idx int64) Move // returns move from neural network encoding output space.

	// Meta-game stuff
	Ended() (ended bool, winner chess.Color) // has the game ended? if yes, then who's the winner?

	// Meta-game stuff
	Score(p chess.Color) float32 // score of the given player

	// interactions
	Check(m Move) bool  // check if the placement is legal
	Apply(m Move) State // should return a GameState. The required side effect is the NextToMove has to change.
	Reset()             // reset state

	// For MCTS
	UndoLastMove()
	Fwd()

	// generics
	Eq(other State) bool
	Clone() State
}
