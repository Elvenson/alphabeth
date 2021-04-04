package mcts

import (
	"context"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/alphabeth/game"
	"github.com/chewxy/math32"
	"github.com/notnil/chess"
)

/*
Here lies the majority of the MCTS search code, while node.go and tree.go handles the data structure stuff.

Right now the code is very specific to the game of Go. Ideally we'd be able to export the correct things and make it
so that a search can be written for any other games but uses the same data structures
*/

const (
	MAXTREESIZE = 25000000 // a tree is at max allowed this many nodes - at about 56 bytes per node that is 1.2GB of memory required
)

// Inferencer is essentially the neural network
type Inferencer interface {
	Infer(state game.State) (policy []float32, value float32)
}

// Result is a NaN tagged floating point, used to represent the reuslts.
type Result float32

const (
	noResultBits = 0x7FE00000
)

func noResult() Result {
	return Result(math32.Float32frombits(noResultBits))
}

// isNullResult returns true if the Result (a NaN tagged number) is noResult
func isNullResult(r Result) bool {
	b := math32.Float32bits(float32(r))
	return b == noResultBits
}

type searchState struct {
	tree          uintptr
	current, prev game.State
	root          naughty
	depth         int

	wg *sync.WaitGroup

	// config
	maxPlayouts, maxVisits, maxDepth int
}

func (s *searchState) nodeCount() int32 {
	t := treeFromUintptr(s.tree)
	return atomic.LoadInt32(&t.nc)
}

func (s *searchState) incrementPlayout() {
	t := treeFromUintptr(s.tree)
	atomic.AddInt32(&t.playouts, 1)
}

func (s *searchState) isRunning() bool {
	t := treeFromUintptr(s.tree)
	running := t.running.Load().(bool)
	return running && t.nodeCount() < MAXTREESIZE
}

func (s *searchState) minPsaRatio() float32 {
	ratio := float32(s.nodeCount()) / float32(MAXTREESIZE)
	switch {
	case ratio > 0.95:
		return 0.01
	case ratio > 0.5:
		return 0.001
	}
	return 0
}

func (t *MCTS) Search() (retVal game.Move) {
	t.updateRoot()
	boardHash := t.current.Hash()

	t.Lock()
	for _, f := range t.freeables {
		t.free(f)
	}
	t.Unlock()

	t.prepareRoot(t.current)
	root := t.nodeFromNaughty(t.root)

	ch := make(chan *searchState, runtime.NumCPU())
	var wg sync.WaitGroup
	for i := 0; i < runtime.NumCPU(); i++ {
		ss := &searchState{
			tree:     ptrFromTree(t),
			current:  t.current,
			root:     t.root,
			maxDepth: t.M * t.N,
			wg:       &wg,
		}
		ch <- ss
	}

	var iter int32
	t.running.Store(true)
	ctx, cancel := context.WithCancel(context.Background())
	for i := 0; i < runtime.NumCPU(); i++ {
		wg.Add(1)
		go doSearch(t.root, &iter, ch, ctx, &wg)
	}
	<-time.After(t.Timeout)
	cancel()

	// TODO
	// reactivate all pruned children
	wg.Wait()
	close(ch)

	root = t.nodeFromNaughty(t.root)
	if !root.HasChildren() {
		policy, _ := t.nn.Infer(t.current)
		moveID := argmax(policy)
		t.log("Returning Early. Best %v", moveID)
		return t.current.NNToMove(int32(moveID))
	}

	retVal = t.current.NNToMove(t.bestMove())
	t.prev = t.current.Clone().(game.State)
	t.log("Move Number %d, Iterations %d Playouts: %v Nodes: %v. Best: %v",
		t.current.MoveNumber(), iter, t.playouts, len(t.nodes), retVal)

	// update the cached policies.
	// Again, nothing like having side effects to what appears to be a straightforwards
	// pure function eh?
	t.cachedPolicies[sa{boardHash, retVal}]++

	return retVal
}

func doSearch(start naughty, iterBudget *int32, ch chan *searchState, ctx context.Context, wg *sync.WaitGroup) {
	defer wg.Done()

loop:
	for {
		select {
		case s := <-ch:
			current := s.current.Clone().(game.State)
			root := start
			res := s.pipeline(current, root)
			if !isNullResult(res) {
				s.incrementPlayout()
			}

			t := treeFromUintptr(s.tree)
			val := atomic.AddInt32(iterBudget, 1)

			if val > t.Budget {
				t.running.Store(false)
			}
			// running := t.running.Load().(bool)
			// running = running && !s.stopThinking( /*TODO*/ )
			// running = running && s.hasAlternateMoves( /*TODO*/ )
			if s.depth == s.maxDepth {
				// reset s for another bout of playouts
				s.root = t.root
				s.current = t.current
				s.depth = 0
			}
			ch <- s
		case <-ctx.Done():
			break loop
		}
	}

	return
}

// pipeline is a recursive MCTS pipeline:
//	SELECT, EXPAND, SIMULATE, BACKPROPAGATE.
//
// Because of the recursive nature, the pipeline is altered a bit to be this:
//	EXPAND and SIMULATE, SELECT and RECURSE, BACKPROPAGATE.
func (s *searchState) pipeline(current game.State, start naughty) (retVal Result) {
	retVal = noResult()
	s.depth++
	if s.depth > s.maxDepth {
		s.depth--
		return
	}
	player := current.Turn()

	// if the game has ended returns negative reward value because we want to return the opposite state
	// from other side perspective
	if ended, winner := current.Ended(); ended {
		if winner == chess.NoColor {
			return 0
		}
		if player == winner {
			return -1
		}
		return 1
	}
	nodeCount := s.nodeCount()

	t := treeFromUintptr(s.tree)
	n := t.nodeFromNaughty(start)
	t.log("\t%p PIPELINE: %v", s, n)

	// EXPAND and SIMULATE
	isExpandable := n.IsExpandable(0)
	if isExpandable && nodeCount < MAXTREESIZE {
		hadChildren := n.HasChildren()
		value, ok := s.expandAndSimulate(start, current, s.minPsaRatio())
		if !hadChildren && ok {
			retVal = Result(value)
		}
	}

	// SELECT and RECURSE
	if n.HasChildren() && isNullResult(retVal) {
		next := t.nodeFromNaughty(n.Select())
		moveIdx := next.Move()
		move := current.NNToMove(moveIdx)
		if current.Check(move) {
			current = current.Apply(move).(game.State)
			retVal = s.pipeline(current, next.id)
		}
	}

	// BACKPROPAGATE
	if !isNullResult(retVal) {
		n.Update(float32(retVal)) // nothing says non functional programs like side effects. Insert more functional programming circle jerk here.
	}
	s.depth--
	return -retVal
}

func (s *searchState) expandAndSimulate(parent naughty, state game.State, minPsaRatio float32) (value float32, ok bool) {
	t := treeFromUintptr(s.tree)
	n := t.nodeFromNaughty(parent)

	t.log("\t\t%p Expand and Simulate. Parent Move: %v. Player: %v. Move number %d\n%v",
		s, n.Move(), state.Turn(), state.MoveNumber(), state)
	if !n.IsExpandable(minPsaRatio) {
		t.log("\t\tNot expandable. MinPSA Ratio %v", minPsaRatio)
		return 0, false
	}

	var policy []float32
	policy, value = t.nn.Infer(state) // get policy probability, value from neural network

	var nodelist []pair
	var legalSum float32

	for i := 0; i < s.current.ActionSpace(); i++ {
		if state.Check(state.NNToMove(int32(i))) {
			nodelist = append(nodelist, pair{Score: policy[i], Move: int32(i)})
			legalSum += policy[i]
		}
	}
	t.log("\t\t%p Available Moves %d: %v", s, len(nodelist), nodelist)

	if legalSum > math32.SmallestNonzeroFloat32 {
		// re normalize
		for i := range nodelist {
			nodelist[i].Score /= legalSum
		}
	} else {
		prob := 1 / float32(len(nodelist))
		for i := range nodelist {
			nodelist[i].Score = prob
		}
	}

	if len(nodelist) == 0 {
		t.log("\t\tNodelist is empty")
		return value, true
	}
	sort.Sort(byScore(nodelist))
	maxPsa := nodelist[0].Score
	oldMinPsa := maxPsa * n.MinPsaRatio()
	newMinPsa := maxPsa * minPsaRatio

	var skippedChildren bool
	for _, p := range nodelist {
		if p.Score < newMinPsa {
			t.log("\t\tp.score %v <  %v", p.Score, newMinPsa)
			skippedChildren = true
		} else if p.Score < oldMinPsa {
			if nn := n.findChild(p.Move); nn == nilNode {
				nn := t.New(p.Move, p.Score, value)
				n.AddChild(nn)
			}
		}
	}
	t.log("\t\t%p skipped children? %v", s, skippedChildren)
	if skippedChildren {
		atomic.StoreUint32(&n.minPSARatioChildren, math32.Float32bits(minPsaRatio))
	} else {
		// if no children were skipped, then all that can be expanded has been expanded
		atomic.StoreUint32(&n.minPSARatioChildren, 0)
	}
	return value, true
}

func (t *MCTS) bestMove() int32 {
	moveNum := t.current.MoveNumber()

	children := t.children[t.root]
	t.log("%p Children: ", &t.searchState)
	for _, child := range children {
		nc := t.nodeFromNaughty(child)
		t.log("\t\t\t%v", nc)
	}
	t.log("%v", t.current)
	t.childLock[t.root].Lock()
	sort.Sort(fancySort{l: children, t: t})
	t.childLock[t.root].Unlock()

	if moveNum < t.Config.RandomCount {
		t.randomizeChildren(t.root)
	}

	// if no children set the current play move to resign and return -1
	if len(children) == 0 {
		t.log("Board\n%v |%v", t.current, t.nodeFromNaughty(t.root))
		t.current.Resign(t.current.Turn())
		return game.Resign
	}

	firstChild := t.nodeFromNaughty(children[0])
	bestMove := firstChild.Move()
	return bestMove
}

func (t *MCTS) prepareRoot(state game.State) {
	root := t.nodeFromNaughty(t.root)
	hadChildren := len(t.children[t.root]) > 0
	expandable := root.IsExpandable(0)
	var value float32
	if expandable {
		value, _ = t.expandAndSimulate(t.root, state, t.minPsaRatio())
	}

	if hadChildren {
		value = root.QSA()
	} else {
		root.Update(value)
	}
}

// newRootState moves the search state to use a new root state. It returns true when a new root state was created.
// As a side effect, the freeables list is also updated.
func (t *MCTS) newRootState() bool {
	if t.root == nilNode || t.prev == nil {
		t.log("No root")
		return false // no current state. Cannot advance to new state
	}
	depth := t.current.MoveNumber() - t.prev.MoveNumber()
	if depth < 0 {
		t.log("depth < 0")
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
	t.log("depth %v", depth)
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

		t.prev = t.prev.Apply(t.prev.NNToMove(move)).(game.State)
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
// If no new root state can be found, a new Node indicating a PASS move is made.
func (t *MCTS) updateRoot() {
	t.freeables = t.freeables[:0]
	if !t.newRootState() || t.searchState.root == nilNode {
		// TODO: Should randomize this instead of for loop.
		for i := 0; i < t.searchState.current.ActionSpace(); i++ {
			if t.searchState.current.Check(t.searchState.current.NNToMove(int32(i))) {
				t.searchState.root = t.New(int32(i), 0, 0)
				break
			}
		}
	}

	t.log("freables %d", len(t.freeables))
	t.searchState.prev = nil
	root := t.nodeFromNaughty(t.searchState.root)
	atomic.StoreInt32(&t.nc, int32(root.countChildren()))

	// if root has no children
	children := t.Children(t.searchState.root)
	if len(children) == 0 {
		atomic.StoreUint32(&root.minPSARatioChildren, defaultMinPsaRatio)
	}

}
