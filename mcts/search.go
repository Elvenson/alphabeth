package mcts

import (
	"fmt"
	"log"
	"math/rand"
	"sort"
	"sync"
	"sync/atomic"

	"github.com/alphabeth/game"
	"github.com/chewxy/math32"
	"github.com/hashicorp/go-multierror"
	"github.com/notnil/chess"
)

const (
	MAXTREESIZE    = 25000000 // a tree is at max allowed this many nodes - at about 56 bytes per node that is 1.2GB of memory required
	epsilon        = 0.25     // For adding Dirichlet noise.
	dirichletParam = 0.3
)

// Inferencer is essentially the neural network
type Inferencer interface {
	Infer(state game.State) (policy []float32, value float32)
}

type searchState struct {
	tree          uintptr
	current, prev game.State
	root          naughty
	wg            *sync.WaitGroup
	maxDepth      int
}

func (s *searchState) nodeCount() int32 {
	t := treeFromUintptr(s.tree)
	return atomic.LoadInt32(&t.nc)
}

// Search using Monte Carlo tree to do simulation and get the best move. Note that we should check for checkmate
// first before running this function
func (t *MCTS) Search() (game.Move, error) {
	t.updateRoot()

	for _, f := range t.freeables {
		t.free(f)
	}

	var eg multierror.Group
	for i := 0; i < t.NumSimulation; i++ {
		eg.Go(func() error {
			g := t.current.Clone()
			_, err := t.pipeline(g, t.root, 0)
			return err
		})
	}
	egErr := eg.Wait()
	if egErr != nil {
		return "", egErr
	}

	root := t.nodeFromNaughty(t.root)
	if !root.HasChildren() {
		return "", fmt.Errorf("no child node in tree")
	}

	t.updatePolicies()
	m, err := t.current.NNToMove(t.bestMove())
	if err != nil {
		return "", err
	}
	t.prev = t.current.Clone().(game.State)

	return m, nil
}

// pipeline is a recursive MCTS pipeline:
// SELECT, EXPAND, SIMULATE, BACKPROPAGATE.
// Because of the recursive nature, the pipeline is altered a bit to be this:
// EXPAND and SIMULATE, SELECT and RECURSE, BACKPROPAGATE.
func (s *searchState) pipeline(current game.State, start naughty, depth int) (float32, error) {
	depth++
	if depth > s.maxDepth {
		log.Printf("reach max depth stop: %d", s.maxDepth)
		return 0, nil
	}
	player := current.Turn()

	// if the game has ended returns negative reward value because we want to return the opposite state
	// from other side perspective
	if ended, winner := current.Ended(); ended {
		if winner == chess.NoColor {
			return 0, nil
		}
		if player == winner {
			return -1, nil
		}
		return 1, nil
	}
	nodeCount := s.nodeCount()
	if nodeCount >= MAXTREESIZE {
		return 0, nil
	}

	t := treeFromUintptr(s.tree)
	n := t.nodeFromNaughty(start)
	hadChildren := n.HasChildren()

	// EXPAND and SIMULATE
	var value float32
	if !hadChildren {
		value, err := s.expandAndSimulate(start, current)
		if err != nil {
			return 0, err
		}
		return -value, nil
	}

	// SELECT and RECURSE
	var next *Node
	next = t.nodeFromNaughty(n.Select())
	moveIdx := next.Move()
	move, err := current.NNToMove(moveIdx)
	if err != nil {
		return 0, err
	}
	current = current.Apply(move).(game.State)
	value, err = s.pipeline(current, next.id, depth)
	if err != nil {
		return 0, err
	}
	next.Update(value)
	return -value, nil
}

// dirichletNoise add Dirichlet noise according to AlphaZero paper
// reference: https://stats.stackexchange.com/questions/322831/purpose-of-dirichlet-noise-in-the-alphazero-paper
func (t *MCTS) dirichletNoise(index int32, p float32) float32 {
	return (1-epsilon)*p + epsilon*float32(t.dirichletSample[index])
}

func (s *searchState) expandAndSimulate(parent naughty, state game.State) (float32, error) {
	t := treeFromUintptr(s.tree)
	n := t.nodeFromNaughty(parent)

	var policy []float32
	policy, value := t.nn.Infer(state) // get policy probability, value from neural network
	if math32.IsNaN(value) {
		log.Printf("nn value is NA returns 0")
		return 0, nil
	}

	var nodelist []pair
	var legalSum float32

	for i := 0; i < s.current.ActionSpace(); i++ {
		m, err := state.NNToMove(int32(i))
		if err != nil {
			return 0, err
		}
		if state.Check(m) {
			nodelist = append(nodelist, pair{Score: policy[i], Move: int32(i)})
			legalSum += policy[i]
		}
	}

	if legalSum > math32.SmallestNonzeroFloat32 {
		for i := range nodelist {
			nodelist[i].Score /= legalSum
			nodelist[i].Score = t.dirichletNoise(nodelist[i].Move, nodelist[i].Score)
		}
	} else {
		prob := 1 / float32(len(nodelist))
		for i := range nodelist {
			nodelist[i].Score = prob
			nodelist[i].Score = t.dirichletNoise(nodelist[i].Move, nodelist[i].Score)
		}
	}

	if len(nodelist) == 0 {
		return value, nil
	}
	sort.Sort(byScore(nodelist))

	for _, p := range nodelist {
		if nn := n.findChild(p.Move); nn == nilNode {
			nn := t.New(p.Move, p.Score)
			n.AddChild(nn)
			n.SetHasChild(true)
		}
	}

	return value, nil
}

func (t *MCTS) bestMove() int32 {
	moveNum := t.current.MoveNumber()

	children := t.children[t.root]
	sort.Sort(fancySort{l: children, t: t})

	idx := 0
	if moveNum < t.Config.RandomCount {
		idx = t.sampleChild()
	}

	// if no children set the current play move to resign and return.
	if len(children) == 0 {
		t.current.Resign(t.current.Turn())
		return game.Resign
	}

	child := t.nodeFromNaughty(children[idx])
	bestMove := child.Move()
	return bestMove
}

// newRootState moves the search state to use a new root state. It returns true when a new root state was created.
// As a side effect, the freeables list is also updated.
func (t *MCTS) newRootState() bool {
	if t.root == nilNode || t.prev == nil {
		return false // no current state. Cannot advance to new state
	}
	depth := t.current.MoveNumber() - t.prev.MoveNumber()
	if depth < 0 {
		return false // oops too far
	}

	tmp := t.current.Clone().(game.State)
	for i := 0; i < depth; i++ {
		tmp.UndoLastMove()
	}
	if !tmp.Eq(t.prev) {
		return false // they're not the same tree - a new root needs to be created
	}
	// try to replay tmp
	for i := 0; i < depth; i++ {
		tmp.Fwd()
		move := tmp.LastMove()
		oldRoot := t.root
		oldRootNode := t.nodeFromNaughty(oldRoot)
		newRoot := oldRootNode.findChild(move)
		if newRoot == nilNode {
			return false
		}
		t.Lock()
		t.root = newRoot
		t.Unlock()
		t.cleanup(oldRoot, newRoot)
		m, err := t.prev.NNToMove(move)
		if err != nil {
			panic(err)
		}
		t.prev = t.prev.Apply(m).(game.State)
	}

	if t.current.MoveNumber() != t.prev.MoveNumber() {
		return false
	}
	if !t.current.Eq(t.prev) {
		return false
	}
	return true
}

// updateRoot updates the root after searching for a new root state.
// If no new root state can be found, a new Node indicating a Begin move is made.
func (t *MCTS) updateRoot() {
	t.freeables = t.freeables[:0]

	// at the beginning of the game make dummy move Begin as a root
	if t.searchState.root == nilNode {
		t.searchState.root = t.New(game.Begin, 0)
	} else if !t.newRootState() { // in the middle of the game, find possible move to continue
		moves := t.current.PossibleMoves()
		randIdx := int32(rand.Intn(len(moves)))
		t.searchState.root = t.New(moves[randIdx], 0)
	}

	t.searchState.prev = nil
	root := t.nodeFromNaughty(t.searchState.root)
	atomic.StoreInt32(&t.nc, int32(root.countChildren()))

	// if root has no children
	children := t.Children(t.searchState.root)
	if len(children) == 0 {
		root.SetHasChild(false)
	}
}

func (t *MCTS) updatePolicies() {
	var denominator float32
	temp := float32(1.0)
	if t.current.MoveNumber() < t.Config.RandomCount {
		temp = t.Config.RandomTemperature
	}
	tree := treeFromUintptr(t.tree)
	children := tree.Children(t.root)
	for _, kid := range children {
		child := tree.nodeFromNaughty(kid)
		if child.IsValid() {
			visits := child.Visits()
			denominator += math32.Pow(float32(visits), 1/t.Config.RandomTemperature)
		}
	}

	policies := make([]float32, t.current.ActionSpace())
	for _, kid := range children {
		child := tree.nodeFromNaughty(kid)
		if child.IsValid() {
			numerator := math32.Pow(float32(child.Visits()), 1/temp)
			p := numerator / denominator
			policies[child.Move()] = p
			child.SetPi(p)
		}
	}
	t.policies = policies
}
