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
		panic(fmt.Errorf("error opening image: %w", err))
	}

	newImage, err := engine.RemoveBackground(img)
	if err != nil {
		panic(fmt.Errorf("error removing background: %w", err))
	}

	newImage, err = engine.SmartCrop(newImage, &rmbg.CropConfig{
		SquareCrop:   true,
		MinThreshold: 200,
	})
	if err != nil {
		panic(fmt.Errorf("error cropping image: %w", err))
	}

	outputPath := "output.jpg"
	err = imaging.Save(newImage, outputPath)
	if err != nil {
		panic(fmt.Errorf("error saving image: %w", err))
	}

}
