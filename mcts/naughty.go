package mcts

// Naughty is essentially *Node
type Naughty int

func (n Naughty) isValid() bool { return n >= 0 }

const (
	nilNode Naughty = -1
)
