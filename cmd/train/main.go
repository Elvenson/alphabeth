package main

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"time"

	agogo "github.com/alphabeth"
	dual "github.com/alphabeth/dualnet"
	"github.com/alphabeth/game"
	"github.com/alphabeth/mcts"
	"github.com/colinmarc/hdfs"
)

var (
	fileMoves = flag.String("moves_file", "", "file containing chess moves")
	modelPath = flag.String("model_path", "alphabeth", "Model checkpoint directory")
)

func compress(src string, buf io.Writer) error {
	// tar > gzip > buf
	zr := gzip.NewWriter(buf)
	tw := tar.NewWriter(zr)

	// walk through every file in the folder
	filepath.Walk(src, func(file string, fi os.FileInfo, err error) error {
		// generate tar header
		header, err := tar.FileInfoHeader(fi, file)
		if err != nil {
			return err
		}

		// must provide real name
		// (see https://golang.org/src/archive/tar/common.go?#L626)
		header.Name = filepath.ToSlash(file)

		// write header
		if err := tw.WriteHeader(header); err != nil {
			return err
		}
		// if not a dir, write file content
		if !fi.IsDir() {
			data, err := os.Open(file)
			if err != nil {
				return err
			}
			if _, err := io.Copy(tw, data); err != nil {
				return err
			}
		}
		return nil
	})

	// produce tar
	if err := tw.Close(); err != nil {
		return err
	}
	// produce gzip
	if err := zr.Close(); err != nil {
		return err
	}
	//
	return nil
}

func compressToTar(modelPath, tarPath string) error {
	// tar + gzip
	var buf bytes.Buffer
	err := compress(modelPath, &buf)
	if err != nil {
		return err
	}

	// write the .tar.gzip
	fileToWrite, err := os.OpenFile(tarPath, os.O_CREATE|os.O_RDWR, os.FileMode(0777))
	if err != nil {
		panic(err)
	}
	if _, err := io.Copy(fileToWrite, &buf); err != nil {
		return err
	}
	return nil
}

func writeToHdfs(tarFile, hdfsPath string) error {
	cli, err := hdfs.NewForUser("hadoop-master-1.deep.shopeemobile.com:8020",
		"ld-bao_bui")
	if err != nil {
		return err
	}

	f, err := cli.Create(hdfsPath)
	if err != nil {
		return err
	}

	b, err := ioutil.ReadFile(tarFile)
	if err != nil {
		return err
	}

	_, err = f.Write(b)
	if err != nil {
		return err
	}
	f.Close()
	return nil
}

func main() {
	flag.Parse()
	log.SetFlags(log.Ltime)
	g := game.ChessGame(*fileMoves)

	conf := agogo.Config{
		Name:            "Alphabeth",
		NNConf:          dual.DefaultConf(game.RowNum, game.ColNum, g.ActionSpace()),
		MCTSConf:        mcts.DefaultConfig(),
		UpdateThreshold: 0.55,
	}

	conf.NNConf.BatchSize = 20
	conf.NNConf.Features = 2 // write a better encoding of the board, and increase features (and that allows you to increase K as well)
	conf.NNConf.K = 3
	conf.NNConf.SharedLayers = 3
	conf.MCTSConf = mcts.Config{
		PUCT:              1.5,
		RandomCount:       10,
		MaxDepth:          10000,
		NumSimulation:     10,
		RandomTemperature: 10,
	}

	conf.Encoder = game.InputEncoder

	a := agogo.New(g, conf)
	log.Print("init done")
	if err := a.LearnAZ(1, 5, 5); err != nil {
		log.Fatalf("error when learning chess: %s", err)
	}

	log.Printf("Save model")
	if err := a.SaveAZ(*modelPath); err != nil {
		log.Fatalf("error when saving model: %s", err)
	}

	log.Printf("Compress model to tar")
	sec := time.Now().Unix()
	tarFile := fmt.Sprintf("model_%d.tar.gz", sec)
	err := compressToTar(*modelPath, tarFile)
	if err != nil {
		log.Fatalf("error when compressing model: %s", err)
	}

	log.Printf("Upload to hdfs")
	err = writeToHdfs(tarFile, fmt.Sprintf("/user/ld-bao_bui/az/model_%d", sec))
	if err != nil {
		log.Fatalf("error when upload model to HDFS: %s", err)
	}

	err = os.RemoveAll(*modelPath)
	if err != nil {
		log.Fatalf("error when remove alphabeth folder: %s", err)
	}
	err = os.RemoveAll(tarFile)
	if err != nil {
		log.Fatalf("error when remove tar model: %s", err)
	}

	fmt.Print("finish saving model\n")
}
