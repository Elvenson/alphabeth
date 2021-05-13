package game

import "github.com/notnil/chess"

// InputEncoder encodes game state to neural input format.
func InputEncoder(g State) []float32 {
	m := g.Board().SquareMap()
	board := make([]float32, RowNum*ColNum)
	for k, v := range m {
		if v == chess.NoPiece {
			board[int8(k)] = 0.001
		} else {
			board[int8(k)] = float32(v)
		}
	}

	playerLayer := make([]float32, RowNum*ColNum)
	next := g.Turn()
	for i := range playerLayer {
		playerLayer[i] = float32(next)
	}
	inputLayer := append(board, playerLayer...)
	return inputLayer
}
