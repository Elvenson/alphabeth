// +build !unsafe

package mcts

// nodeFromNaughty gets the node given the pointer.
func (t *MCTS) nodeFromNaughty(ptr Naughty) *Node {
	t.RLock()
	defer t.RUnlock()
	nodes := t.nodes
	retVal := &nodes[int(ptr)]
	return retVal
}

// Children returns a list of children
func (t *MCTS) Children(of Naughty) []Naughty {
	t.RLock()
	defer t.RUnlock()
	retVal := t.children[of]
	return retVal
}
