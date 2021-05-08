package main

import (
	"flag"
	"fmt"

	agogo "github.com/alphabeth"
	"github.com/alphabeth/game"
)

var (
	fileMoves = flag.String("moves_file", "", "file containing chess moves")
	dirName = flag.String("model_path", "", "directory contains trained model")
)

func main() {
	flag.Parse()
	az, err := agogo.Load(*dirName, *fileMoves, game.InputEncoder)
	if err != nil {
		fmt.Printf("error loading model: %s\n", err)
	}
	fmt.Printf("model name is %s\n", az.Name())
}
