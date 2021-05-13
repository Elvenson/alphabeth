package mcts

import (
	"github.com/chewxy/math32"
)

// fancySort sorts the list of nodes under a certain condition of evaluation (i.e. which colour are we considering)
// it sorts in such a way that nils get put at the back
type fancySort struct {
	l []naughty
	t *MCTS
}

func (l fancySort) Len() int      { return len(l.l) }
func (l fancySort) Swap(i, j int) { l.l[i], l.l[j] = l.l[j], l.l[i] }
func (l fancySort) Less(i, j int) bool {
	li := l.t.nodeFromNaughty(l.l[i])
	lj := l.t.nodeFromNaughty(l.l[j])

	return li.Pi() > lj.Pi()
}

// pair is a tuple of score and coordinate
type pair struct {
	Move  int32
	Score float32
}

// byScore is a sortable list of pairs It sorts the list with best score fist
type byScore []pair

func (l byScore) Len() int           { return len(l) }
func (l byScore) Less(i, j int) bool { return l[i].Score > l[j].Score }
func (l byScore) Swap(i, j int)      { l[i], l[j] = l[j], l[i] }

func argmax(a []float32) int {
	var retVal int
	var max = math32.Inf(-1)
	for i := range a {
		if a[i] > max {
			max = a[i]
			retVal = i
		}
	}
	return retVal
}
