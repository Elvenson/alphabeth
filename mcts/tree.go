package mcts

import (
	"math/rand"
	"runtime"
	"sync"
	"time"

	"github.com/alphabeth/game"
	"github.com/chewxy/math32"
)

// Config is the structure to configure the MCTS multitree (poorly named Tree)
type Config struct {
	// PUCT is the proportion of polynomial upper confidence trees to keep. Between 1 and 0
	PUCT    float32
	Timeout time.Duration

	RandomCount       int   // if the move number is less than this, we should randomize
	Budget            int32 // iteration budget
	RandomMinVisits   uint32
	RandomTemperature float32
	MaxDepth          int
	NumSimulation     int // Be careful with this config it can cause goroutine starvation.
}

func DefaultConfig() Config {
	return Config{
		PUCT:    1.0,
		Timeout: 100 * time.Millisecond,
		Budget:  10000,
	}
}

func (c Config) IsValid() bool {
	return c.PUCT > 0 && c.PUCT <= 1
}

// sa is a state-action tuple, used for storing results
type sa struct {
	s [16]byte
	a game.Move
}

// MCTS is essentially a "global" manager of sorts for the memories. The goal is to build MCTS without much pointer chasing.
type MCTS struct {
	sync.RWMutex
	Config
	nn   Inferencer
	rand *rand.Rand

	// memory related fields
	nodes []Node
	// children  map[naughty][]naughty
	children [][]naughty

	freelist  []naughty
	freeables []naughty // list of nodes that can be freed

	// global searchState
	searchState
	nc int32 // atomic pls

	// global policy values - useful for building policy vectors
	cachedPolicies map[sa]float32

	lumberjack
}

func New(game game.State, conf Config, nn Inferencer) *MCTS {
	retVal := &MCTS{
		Config: conf,
		nn:     nn,
		rand:   rand.New(rand.NewSource(time.Now().UnixNano())),

		nodes: make([]Node, 0, 12288),
		// children: make(map[naughty][]naughty),
		children:  make([][]naughty, 0, 12288),

		searchState: searchState{
			root:    nilNode,
			current: game,
		},

		cachedPolicies: make(map[sa]float32),
		lumberjack:     makeLumberJack(),
	}
	go retVal.start()
	retVal.searchState.tree = ptrFromTree(retVal)
	retVal.searchState.maxDepth = conf.MaxDepth
	return retVal
}

// New creates a new node
func (t *MCTS) New(move int32, score float32) (retVal naughty) {
	n := t.alloc()
	N := t.nodeFromNaughty(n)
	N.lock.Lock()
	defer N.lock.Unlock()
	N.move = move
	N.visits = 1
	N.status = uint32(Active)
	N.qsa = 0
	N.psa = score

	return n
}

// SetGame sets the game
func (t *MCTS) SetGame(g game.State) {
	t.Lock()
	t.current = g
	t.Unlock()
}

func (t *MCTS) Nodes() int { return len(t.nodes) }

func (t *MCTS) Policies(g game.State) ([]float32, error) {
	hash := g.Hash()
	var sum float32
	retVal := make([]float32, g.ActionSpace())
	for i := 0; i < g.ActionSpace(); i++ {
		m, err := t.current.NNToMove(int32(i))
		if err != nil {
			return nil, err
		}
		if t.current.Check(m) {
			prob := t.cachedPolicies[sa{s: hash, a: m}]
			retVal[i] = prob
			sum += prob
		}
	}
	for i := 0; i < len(retVal); i++ {
		retVal[i] /= sum
	}
	return retVal, nil
}

// alloc tries to get a node from the free list. If none is found a new node is allocated into the master arena
func (t *MCTS) alloc() naughty {
	t.Lock()
	defer t.Unlock()
	l := len(t.freelist)
	if l == 0 {
		N := Node{
			lock:        sync.Mutex{},
			tree:        ptrFromTree(t),
			id:          naughty(len(t.nodes)),
			hasChildren: false,
		}
		t.nodes = append(t.nodes, N)
		t.children = append(t.children, make([]naughty, 0, t.current.ActionSpace()))
		n := naughty(len(t.nodes) - 1)
		return n
	}

	i := t.freelist[l-1]
	t.freelist = t.freelist[:l-1]
	return i
}

// free puts the node back into the freelist.
//
// Because the there isn't really strong reference tracking, there may be
// use-after-free issues. Therefore it's absolutely vital that any calls to free()
// has to be done with careful consideration.
func (t *MCTS) free(n naughty) {
	// delete(t.children, n)
	t.children[int(n)] = t.children[int(n)][:0]
	t.freelist = append(t.freelist, n)
	N := &t.nodes[int(n)]
	N.reset()
}

// cleanup cleans up the graph (WORK IN PROGRESS)
func (t *MCTS) cleanup(oldRoot, newRoot naughty) {
	children := t.Children(oldRoot)
	// we aint going down other paths, those nodes can be freed
	for _, kid := range children {
		if kid != newRoot {
			t.nodeFromNaughty(kid).Invalidate()
			t.freeables = append(t.freeables, kid)
			t.cleanChildren(kid)
		}
	}
	t.Lock()
	t.children[oldRoot] = t.children[oldRoot][:1]
	t.children[oldRoot][0] = newRoot
	t.Unlock()
}

func (t *MCTS) cleanChildren(root naughty) {
	children := t.Children(root)
	for _, kid := range children {
		t.nodeFromNaughty(kid).Invalidate()
		t.freeables = append(t.freeables, kid)
		t.cleanChildren(kid) // recursively clean children
	}
	t.Lock()
	t.children[root] = t.children[root][:0] // empty it
	t.Unlock()
}

// randomizeChildren proportionally randomizes the children nodes by proportion of the visit
func (t *MCTS) randomizeChildren(of naughty) {
	var accum, norm float32
	var accumVector []float32
	children := t.Children(of)
	for _, kid := range children {
		child := t.nodeFromNaughty(kid)
		visits := child.Visits()
		if norm == 0 {
			norm = float32(visits)

			// nonsensical options
			if visits <= t.Config.RandomMinVisits {
				return
			}
		}
		if visits > t.Config.RandomMinVisits {
			accum += math32.Pow(float32(visits)/norm, 1/t.Config.RandomTemperature)
			accumVector = append(accumVector, accum)
		}
	}
	rnd := t.rand.Float32() * accum // uniform distro: rnd() * (max-min) + min
	var index int
	for i, a := range accumVector {
		if rnd < a {
			index = i
			break
		}
	}
	if index == 0 {
		return
	}

	children[0], children[index] = children[index], children[0]

}

func (t *MCTS) Reset() {
	t.Lock()
	defer t.Unlock()

	t.freelist = t.freelist[:0]
	t.freeables = t.freeables[:0]
	for i := range t.nodes {
		t.nodes[i].move = -1
		t.nodes[i].visits = 0
		t.nodes[i].status = 0
		t.nodes[i].psa = 0
		t.nodes[i].hasChildren = false
		t.nodes[i].qsa = 0
		t.freelist = append(t.freelist, t.nodes[i].id)
	}

	for i := range t.children {
		t.children[i] = t.children[i][:0]
	}

	t.nodes = t.nodes[:0]
	t.cachedPolicies = make(map[sa]float32)
	runtime.GC()
}
