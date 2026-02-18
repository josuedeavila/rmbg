package rmbg

import (
	"image"
	"image/color"
	"os"
	"path/filepath"
	"testing"
)

func TestRemBG_Integration(t *testing.T) {
	// Path to the model used in the example
	modelPath := filepath.Join("example", "models", "u2netp.onnx")

	// Skip if model file doesn't exist (e.g. in CI environments without the binary model)
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skipf("Skipping integration test: model not found at %s", modelPath)
	}

	config := &Config{
		ModelPath:         modelPath,
		IntraOpNumThreads: 1,
		InterOpNumThreads: 1,
		CpuMemArena:       true,
		MemPattern:        true,
	}

	remover, err := New(config)
	if err != nil {
		t.Fatalf("Failed to create RemBG: %v", err)
	}
	defer remover.Close()

	// Create a simple test image (a white square on a black background)
	img := image.NewRGBA(image.Rect(0, 0, 100, 100))
	for y := 0; y < 100; y++ {
		for x := 0; x < 100; x++ {
			if x > 25 && x < 75 && y > 25 && y < 75 {
				img.Set(x, y, color.White)
			} else {
				img.Set(x, y, color.Black)
			}
		}
	}

	t.Run("RemoveBackground", func(t *testing.T) {
		out, err := remover.RemoveBackground(img)
		if err != nil {
			t.Errorf("RemoveBackground failed: %v", err)
		}
		if out == nil {
			t.Error("Expected output image, got nil")
		}
	})

	t.Run("SmartCrop", func(t *testing.T) {
		cropConfig := &CropConfig{
			Margin:       5,
			MinThreshold: 10,
		}
		cropped, err := remover.SmartCrop(img, cropConfig)
		if err != nil {
			t.Errorf("SmartCrop failed: %v", err)
		}
		if cropped == nil {
			t.Error("Expected cropped image, got nil")
		}
	})
}
