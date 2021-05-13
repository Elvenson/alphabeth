# Alphabeth

AlphaZero implementation in Golang inspired by [gorgonia/agogo](https://github.com/gorgonia/agogo) library 
but focus only on Chess game. For more information, you can take a look at this [paper](https://arxiv.org/pdf/1712.01815.pdf).

## Installation
To install package, just simply type:
```shell script
go get github.com/Elvenson/alphabeth
```

## Quickstart
### Training
To train the model, you can take a look at `cmd/train` folder. First we need to compile it by running the following
commands:
```shell script
cd cmd/train; go build
```

You will see `train` binary inside your `cmd/train` folder. To run it simply type:
```shell script
./train -moves_file=chess_moves.txt -model_path=alphabet
```
You will see model checkpoint in newly created folder named `alphabet` as specified in your command parameters.

**Note**: This is just an example on how to train a model, in order to train a better model you should tune your
parameters as well as writing a better feature generation part in `game/encoding.go` script.

### Inference
To test inference part, simply run the following commands:
```shell script
cd cmd/infer; go build
```

You can run inference path by typing this command:
```shell script
./infer -moves_file=../train/chess_moves.txt -model_path=../train/alphabeth/
```
The expected output is `model name is Alphabeth`.

### Move generation
As model needs to output a vector with dimension corresponding to the total possible moves in Chess game. According to
the paper, this number is `4,672` possible moves. But in this implementation, we will only get the subset of possible moves
which are frequently played by randomly playing a number of games to generate number of possible moves.

To run this script, you can run the following commands:
```shell script
cd cmd/generatemoves; go build
```

then type:
```shell script
./generatemoves -num_game=10 -path=chess_moves.txt
```
A file named `chess_moves.txt` will be created which contains legal possible moves in each line in UCI notation.

**Note**: train and inference parts should pass the same chess moves file via `--moves_file` parameter.

## Contribution
Any contribution is greatly appreciated since some of the detail in AlphaZero paper has not been implemented 
or can be misunderstood by me.