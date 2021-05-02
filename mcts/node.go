package mcts

import (
	"fmt"
	"sync"

	"github.com/chewxy/math32"
)

type Status uint32

const (
	Invalid Status = iota
	Active
	Pruned
)

func (a Status) String() string {
	switch a {
	case Invalid:
		return "Invalid"
	case Active:
		return "Active"
	case Pruned:
		return "Pruned"
	}
	return "UNKNOWN STATUS"
}

type Node struct {
	// should guarantee thread-safe operation
	lock        sync.Mutex
	move        int32   // should be neural output index
	visits      uint32  // visits to this node - N(s, a) in the literature
	status      uint32  // status
	qsa         float32 // the expected reward for taking action a from state s, i.e: Q(s,a)
	hasChildren bool
	psa         float32 // Neural network policy estimation for taking the move from state s, i.e: P(s, a)

	// naughty things
	id   naughty // index to the children allocation
	tree uintptr // pointer to the tree
}

func (n *Node) Format(s fmt.State, c rune) {
	fmt.Fprintf(s, "{NodeID: %v, Move: %v,"+
		" Q(s,a) %v, P(s,a) %v, Visits %v, Status: %v}", n.id, n.Move(), n.QSA(), n.PSA(),
		n.Visits(), Status(n.status))
}

// AddChild adds a child to the node
func (n *Node) AddChild(child naughty) {
	tree := treeFromUintptr(n.tree)
	tree.Lock()
	tree.children[n.id] = append(tree.children[n.id], child)
	tree.Unlock()
}

// IsFirstVisit returns true if this node hasn't ever been visited
func (n *Node) IsNotVisited() bool {
	n.lock.Lock()
	defer n.lock.Unlock()
	visits := n.visits
	return visits == 0
}

// Update updates the accumulated score
func (n *Node) Update(score float32) {
	n.accumulate(score)
	n.lock.Lock()
	n.visits += 1
	n.lock.Unlock()
}

// QSA returns Q(s, a)
func (n *Node) QSA() float32 {
	n.lock.Lock()
	defer n.lock.Unlock()
	v := n.qsa
	return v
}

// Move gets the move associated with the node
func (n *Node) Move() int32 {
	n.lock.Lock()
	defer n.lock.Unlock()
	m := n.move
	return m
}

// PSA returns P(s, a)
func (n *Node) PSA() float32 {
	n.lock.Lock()
	defer n.lock.Unlock()
	v := n.psa
	return v
}

func (n *Node) Visits() uint32 {
	n.lock.Lock()
	defer n.lock.Unlock()
	v := n.visits
	return v
}

// Activate activates the node
func (n *Node) Activate() {
	n.lock.Lock()
	defer n.lock.Unlock()
	n.status = uint32(Active)
}

// Prune prunes the node
func (n *Node) Prune() {
	n.lock.Lock()
	defer n.lock.Unlock()
	n.status = uint32(Pruned)
}

// Invalidate invalidates the node
func (n *Node) Invalidate() {
	n.lock.Lock()
	defer n.lock.Unlock()
	n.status = uint32(Invalid)
}

// IsValid returns true if it's valid
func (n *Node) IsValid() bool {
	n.lock.Lock()
	defer n.lock.Unlock()
	status := n.status
	return Status(status) != Invalid
}

// IsActive returns true if the node is active
func (n *Node) IsActive() bool {
	n.lock.Lock()
	defer n.lock.Unlock()
	status := n.status
	return Status(status) == Active
}

// IsPruned returns true if the node has been pruned.
func (n *Node) IsPruned() bool {
	n.lock.Lock()
	defer n.lock.Unlock()
	status := n.status
	return Status(status) == Pruned
}

// HasChildren returns true if the node has children
func (n *Node) HasChildren() bool {
	n.lock.Lock()
	defer n.lock.Unlock()
	return n.hasChildren
}

func (n *Node) SetHasChild(f bool) {
	n.lock.Lock()
	defer n.lock.Unlock()
	n.hasChildren = f
}

func (n *Node) ID() int { return int(n.id) }

// Select selects the best child based on alpha zero paper
// the upper bound formula is as such
// U(s, a) = Q(s, a) + tree.PUCT * P(s, a) * ((sqrt(parent visits))/ (1+visits to this node))
//
// where
// U(s, a) = upper confidence bound given state and action
// Q(s, a) = reward of taking the action given the state
// P(s, a) = initial probability/estimate of taking an action from the state given according to the policy
//
// in the following code,
// psa = P(s, a)
// qsa = Q(s, a)
//
// Given the state and action is already known and encoded into Node itself,it doesn't have to be a function
// like in most MCTS tutorials. This allows it to be slightly more performant (i.e. a AoS-ish data structure)
func (n *Node) Select() naughty {
	var parentVisits uint32

	tree := treeFromUintptr(n.tree)
	children := tree.Children(n.id)
	for _, kid := range children {
		child := tree.nodeFromNaughty(kid)
		if child.IsValid() {
			visits := child.Visits()
			parentVisits += visits
		}
	}

	best := nilNode
	var bestValue = math32.Inf(-1)
	numerator := math32.Sqrt(float32(parentVisits))

	for _, kid := range children {
		child := tree.nodeFromNaughty(kid)
		if !child.IsActive() {
			continue
		}

		qsa := float32(0)
		visits := child.Visits()
		if visits > 0 {
			qsa = child.QSA() // but if this node has been visited before, Q from the node is used.
		}
		psa := child.PSA()
		denominator := 1.0 + float32(visits)
		lastTerm := numerator / denominator
		puct := tree.PUCT * psa * lastTerm
		usa := qsa + puct

		if usa > bestValue {
			bestValue = usa
			best = kid
		}
	}

	if best == nilNode {
		panic("Cannot return nil")
	}
	return best
}

// accumulate updates Q(s, a) thread-safe.
func (n *Node) accumulate(v float32) {
	n.lock.Lock()
	defer n.lock.Unlock()
	qsa := (float32(n.visits)*n.qsa + v) / float32(n.visits+1)
	n.qsa = qsa
}

// countChildren counts the number of children node a node has and number of grandkids recursively
func (n *Node) countChildren() (retVal int) {
	tree := treeFromUintptr(n.tree)
	children := tree.Children(n.id)
	for _, kid := range children {
		child := tree.nodeFromNaughty(kid)
		if child.IsActive() {
			retVal += child.countChildren()
		}
		retVal++ // plus the child itself
	}
	return
}

// findChild finds the first child that has the wanted move
func (n *Node) findChild(move int32) naughty {
	tree := treeFromUintptr(n.tree)
	children := tree.Children(n.id)
	for _, kid := range children {
		child := tree.nodeFromNaughty(kid)
		if child.Move() == move {
			return kid
		}
	}
	return nilNode
}

func (n *Node) reset() {
	n.lock.Lock()
	defer n.lock.Unlock()
	n.move = -1
	n.visits = 0
	n.status = 0
	n.qsa = 0
	n.hasChildren = false
	n.psa = 0
}
