package main

import (
	"fmt"
	"time"

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

	start := time.Now()
	newImage, err := engine.RemoveBackground(img)
	if err != nil {
		panic(fmt.Errorf("error cropping image: %w", err))
	}
	fmt.Printf("time for removing background: %v\n", time.Since(start))

	start = time.Now()
	copped, err := engine.SmartCropFromMask(newImage, rmbg.AutoMask, &rmbg.CropConfig{
		Margin:       10,
		SquareCrop:   true,
		MinThreshold: 100,
	})
	if err != nil {
		panic(fmt.Errorf("error cropping image: %w", err))
	}
	fmt.Printf("time for cropping image: %v\n", time.Since(start))

	outputPath := "output.jpg"
	err = imaging.Save(copped, outputPath)
	if err != nil {
		panic(fmt.Errorf("error saving image: %w", err))
	}

}
