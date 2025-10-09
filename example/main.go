package main

import (
	"fmt"

	"github.com/disintegration/imaging"
	"github.com/josuedeavila/rmbg"
)

func main() {
	cfg := &rmbg.Config{
		IntraOpNumThreads: 1,
		InterOpNumThreads: 1,
		CpuMemArena:       false,
		MemPattern:        true,
		ModelPath:         "./models/u2netp.onnx",
	}
	engine, err := rmbg.New(cfg)
	if err != nil {
		panic(err)
	}
	defer engine.Close()

	inputPath := "input.jpg"
	img, err := imaging.Open(inputPath)
	if err != nil {
		panic(fmt.Errorf("erro ao abrir imagem: %w", err))
	}

	newImage, err := engine.RemoveBackground(img)
	if err != nil {
		panic(fmt.Errorf("erro ao remover fundo: %w", err))
	}

	outputPath := "output.jpg"
	err = imaging.Save(newImage, outputPath)
	if err != nil {
		panic(fmt.Errorf("erro ao salvar imagem: %w", err))
	}

}
