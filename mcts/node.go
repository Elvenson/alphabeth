package mcts

import (
	"fmt"
	"sync/atomic"

	"github.com/chewxy/math32"
)

type Status uint32

const (
	Invalid Status = iota
	Active
	Pruned
)

const (
	defaultMinPsaRatio = 0x40000000 // 2 in float32
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
	// atomic access only pls
	move                int32  // should be neural output index
	visits              uint32 // visits to this node - N(s, a) in the literature
	status              uint32 // status
	qsa                 uint32 // actually float32, the expected reward for taking action a from state s, i.e: Q(s,a)
	minPSARatioChildren uint32 // actually float32. minimum P(s,a) ratio for the children. Default to 2
	psa                 uint32 // Neural network policy estimation for taking the move from state s, i.e: P(s, a)
	value               uint32 // value from the neural network

	// naughty things
	id   naughty // index to the children allocation
	tree uintptr // pointer to the tree
}

func (n *Node) Format(s fmt.State, c rune) {
	fmt.Fprintf(s, "{NodeID: %v Move: %v, Score: %v,"+
		" Q(s,a) %v Visits %v minPSARatioChildren %v Status: %v}", n.id, n.Move(), n.QSA(), n.value,
		n.Visits(), n.minPSARatioChildren, Status(n.status))
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
	visits := atomic.LoadUint32(&n.visits)
	return visits == 0
}

// Update updates the accumulated score
func (n *Node) Update(score float32) {
	t := treeFromUintptr(n.tree)
	t.Lock()
	n.accumulate(score)
	atomic.AddUint32(&n.visits, 1)
	t.Unlock()
}

// QSA returns Q(s, a)
func (n *Node) QSA() float32 {
	score := atomic.LoadUint32(&n.qsa)
	return math32.Float32frombits(score)
}

// Move gets the move associated with the node
func (n *Node) Move() int32 { return atomic.LoadInt32(&n.move) }

// PSA returns P(s, a)
func (n *Node) PSA() float32 {
	v := atomic.LoadUint32(&n.psa)
	return math32.Float32frombits(v)
}

// Value returns the predicted value (probability of winning from the NN) of the given node
func (n *Node) Value() float32 {
	v := atomic.LoadUint32(&n.value)
	return math32.Float32frombits(v)
}

func (n *Node) Visits() uint32 { return atomic.LoadUint32(&n.visits) }

// Activate activates the node
func (n *Node) Activate() { atomic.StoreUint32(&n.status, uint32(Active)) }

// Prune prunes the node
func (n *Node) Prune() { atomic.StoreUint32(&n.status, uint32(Pruned)) }

// Invalidate invalidates the node
func (n *Node) Invalidate() { atomic.StoreUint32(&n.status, uint32(Invalid)) }

// IsValid returns true if it's valid
func (n *Node) IsValid() bool {
	status := atomic.LoadUint32(&n.status)
	return Status(status) != Invalid
}

// IsActive returns true if the node is active
func (n *Node) IsActive() bool {
	status := atomic.LoadUint32(&n.status)
	return Status(status) == Active
}

// IsPruned returns true if the node has been pruned.
func (n *Node) IsPruned() bool {
	status := atomic.LoadUint32(&n.status)
	return Status(status) == Pruned
}

// HasChildren returns true if the node has children
func (n *Node) HasChildren() bool { return n.MinPsaRatio() <= 1 }

// IsExpandable returns true if the node is expandable. It may not be for memory reasons.
func (n *Node) IsExpandable(minPsaRatio float32) bool { return minPsaRatio < n.MinPsaRatio() }

func (n *Node) MinPsaRatio() float32 {
	v := atomic.LoadUint32(&n.minPSARatioChildren)
	return math32.Float32frombits(v)
}

func (n *Node) ID() int { return int(n.id) }

// NNEvaluate returns the result of the NN evaluation of the colour.
func (n *Node) NNEvaluate() float32 {
	return n.Value()
}

// Select selects the child of the given Colour
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

	var best naughty
	var bestValue = math32.Inf(-1)
	fpu := n.NNEvaluate()                           // first play urgency is the value predicted by the NN
	numerator := math32.Sqrt(float32(parentVisits)) // in order to find the stochastic policy, we need to normalize the count

	for _, kid := range children {
		child := tree.nodeFromNaughty(kid)
		if !child.IsActive() {
			continue
		}

		qsa := fpu // the initial Q is what the NN predicts
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

// accumulate updates Q(s, a) atomically.
func (n *Node) accumulate(v float32) {
	b := atomic.LoadUint32(&n.qsa)
	qsa := math32.Float32frombits(b)
	qsa = (float32(n.Visits())*qsa + v) / float32(n.Visits()+1)
	b = math32.Float32bits(qsa)
	atomic.StoreUint32(&n.qsa, b)
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
	atomic.StoreInt32(&n.move, -1)
	atomic.StoreUint32(&n.visits, 0)
	atomic.StoreUint32(&n.status, 0)
	atomic.StoreUint32(&n.qsa, 0)
	atomic.StoreUint32(&n.minPSARatioChildren, defaultMinPsaRatio)
	atomic.StoreUint32(&n.psa, 0)
	atomic.StoreUint32(&n.value, 0)
}
