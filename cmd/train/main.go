package main

import (
	"flag"
	"fmt"

	"github.com/alphabeth/game"
)

var(
	fileMoves = flag.String("moves_file", "", "file containing chess moves")
)

func main() {
	flag.Parse()
	g := game.ChessGame(*fileMoves)
	fmt.Printf("action space is %d", g.ActionSpace())
}
