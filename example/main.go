package main

import (
	"fmt"
	"log"

	bgrm "github.com/josuedeavila/rmbg"
)

func main() {
	// Create a new RemBG instance
	// The model will be downloaded to ./models directory
	remover, err := bgrm.New("./models")
	if err != nil {
		log.Fatal(err)
	}
	defer remover.Close()

	// Remove background from a file
	inputPath := "input.jpg"
	outputPath := "output.png"

	fmt.Printf("Processing %s...\n", inputPath)

	err = remover.RemoveBackgroundFromFile(inputPath, outputPath)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Background removed! Saved to %s\n", outputPath)
}
