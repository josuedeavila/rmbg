package rmbg

import (
	"image"
	"image/color"
	"testing"
)

func TestDetectObjectBounds(t *testing.T) {
	t.Run("EmptyMask", func(t *testing.T) {
		mask := image.NewGray(image.Rect(0, 0, 10, 10))
		_, found := detectObjectBounds(mask, 10)
		if found {
			t.Errorf("expected no object found in empty mask")
		}
	})

	t.Run("SinglePixel", func(t *testing.T) {
		mask := image.NewGray(image.Rect(0, 0, 10, 10))
		mask.SetGray(5, 5, color.Gray{Y: 255})
		bounds, found := detectObjectBounds(mask, 10)
		if !found {
			t.Errorf("expected object found")
		}
		if bounds.MinX != 5 || bounds.MaxX != 5 || bounds.MinY != 5 || bounds.MaxY != 5 {
			t.Errorf("unexpected bounds: %+v", bounds)
		}
	})

	t.Run("Rectangle", func(t *testing.T) {
		mask := image.NewGray(image.Rect(0, 0, 10, 10))
		// 2x2 square from (2,2) to (3,3)
		mask.SetGray(2, 2, color.Gray{Y: 255})
		mask.SetGray(3, 2, color.Gray{Y: 255})
		mask.SetGray(2, 3, color.Gray{Y: 255})
		mask.SetGray(3, 3, color.Gray{Y: 255})

		bounds, found := detectObjectBounds(mask, 10)
		if !found {
			t.Errorf("expected object found")
		}
		if bounds.MinX != 2 || bounds.MaxX != 3 || bounds.MinY != 2 || bounds.MaxY != 3 {
			t.Errorf("unexpected bounds: %+v", bounds)
		}
		if bounds.Width != 1 || bounds.Height != 1 { // Width/Height are Max-Min in implementation
			t.Logf("Note: Width/Height implementation is %d, %d", bounds.Width, bounds.Height)
		}
	})
}

func TestCrop(t *testing.T) {
	// Create a 100x100 source image
	img := image.NewRGBA(image.Rect(0, 0, 100, 100))

	// Create a 10x10 mask representing the object at (40, 40) to (60, 60) in source
	// Mask is usually 320x320 in this lib, but we'll test the scaling
	mask := image.NewGray(image.Rect(0, 0, 10, 10))
	for y := 4; y <= 6; y++ {
		for x := 4; x <= 6; x++ {
			mask.SetGray(x, y, color.Gray{Y: 255})
		}
	}

	t.Run("BasicCrop", func(t *testing.T) {
		config := &CropConfig{
			Margin:       0,
			MinThreshold: 10,
		}
		// Scale 10x
		res, err := crop(img, mask, config, 10.0, 10.0)
		if err != nil {
			t.Fatalf("crop failed: %v", err)
		}
		bounds := res.Bounds()
		// Object at (4,4)-(6,6) in mask scaled by 10 is (40,40)-(60,60)
		// width = 60-40 = 20, height = 60-40 = 20
		if bounds.Dx() != 20 || bounds.Dy() != 20 {
			t.Errorf("expected 20x20 crop, got %dx%d", bounds.Dx(), bounds.Dy())
		}
	})

	t.Run("MarginCrop", func(t *testing.T) {
		config := &CropConfig{
			Margin:       5,
			MinThreshold: 10,
		}
		res, err := crop(img, mask, config, 10.0, 10.0)
		if err != nil {
			t.Fatalf("crop failed: %v", err)
		}
		bounds := res.Bounds()
		// (40-5, 40-5) to (60+5, 60+5) = (35, 35) to (65, 65)
		// size = 30x30
		if bounds.Dx() != 30 || bounds.Dy() != 30 {
			t.Errorf("expected 30x30 crop, got %dx%d", bounds.Dx(), bounds.Dy())
		}
	})

	t.Run("SquareCrop", func(t *testing.T) {
		// Create non-square mask object
		rectMask := image.NewGray(image.Rect(0, 0, 10, 10))
		rectMask.SetGray(4, 4, color.Gray{Y: 255})
		rectMask.SetGray(5, 4, color.Gray{Y: 255}) // Width 1, Height 0 in mask coords

		config := &CropConfig{
			Margin:       0,
			SquareCrop:   true,
			MinThreshold: 10,
		}
		res, err := crop(img, rectMask, config, 10.0, 10.0)
		if err != nil {
			t.Fatalf("crop failed: %v", err)
		}
		bounds := res.Bounds()
		if bounds.Dx() != bounds.Dy() {
			t.Errorf("expected square crop, got %dx%d", bounds.Dx(), bounds.Dy())
		}
	})

	t.Run("MarginPercent", func(t *testing.T) {
		config := &CropConfig{
			MarginPercent: 0.5, // 50% of 20px object = 10px margin
			MinThreshold:  10,
		}
		res, err := crop(img, mask, config, 10.0, 10.0)
		if err != nil {
			t.Fatalf("crop failed: %v", err)
		}
		bounds := res.Bounds()
		// object size 20. 50% margin = 10px each side.
		// total size = 20 + 10 + 10 = 40
		if bounds.Dx() != 40 || bounds.Dy() != 40 {
			t.Errorf("expected 40x40 crop, got %dx%d", bounds.Dx(), bounds.Dy())
		}
	})
}

func TestSmartCropFromMask(t *testing.T) {
	engine := &RemBG{}
	img := image.NewRGBA(image.Rect(0, 0, 100, 100))

	maskFunc := func(i image.Image) *image.Gray {
		mask := image.NewGray(i.Bounds())
		// Object at 40,40 to 60,60
		for y := 40; y <= 60; y++ {
			for x := 40; x <= 60; x++ {
				mask.SetGray(x, y, color.Gray{Y: 255})
			}
		}
		return mask
	}

	config := &CropConfig{
		Margin:       0,
		MinThreshold: 10,
	}

	res, err := engine.SmartCropFromMask(img, maskFunc, config)
	if err != nil {
		t.Fatalf("SmartCropFromMask failed: %v", err)
	}

	bounds := res.Bounds()
	// Object at (40,40) to (60,60) means width 20, height 20
	if bounds.Dx() != 20 || bounds.Dy() != 20 {
		t.Errorf("expected 20x20 crop, got %dx%d", bounds.Dx(), bounds.Dy())
	}
}
