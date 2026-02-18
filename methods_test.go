package rmbg

import (
	"image"
	"image/color"
	"testing"
)

func TestRemBG_Close(t *testing.T) {
	// This test depends on ONNX Runtime being initialized.
	// Since New() creates a session, we'll use a mockable approach or
	// rely on the fact that if New succeeded, Close should too.
	// However, we can't easily mock ort.DynamicAdvancedSession.
	// We'll skip if environment isn't ready.

	config := &Config{
		ModelPath: "example/models/u2netp.onnx",
	}

	r, err := New(config)
	if err != nil {
		t.Skip("Skipping Close test: session could not be created")
		return
	}

	err = r.Close()
	if err != nil {
		t.Errorf("Close() returned error: %v", err)
	}
}

func TestBlendParallel(t *testing.T) {
	bounds := image.Rect(0, 0, 10, 10)
	dst := image.NewRGBA(bounds)

	// Red source image
	src := image.NewRGBA(bounds)
	for i := 0; i < len(src.Pix); i += 4 {
		src.Pix[i] = 255   // R
		src.Pix[i+3] = 255 // A
	}

	// Mask: half transparent, half opaque
	mask := image.NewGray(bounds)
	for y := 0; y < 10; y++ {
		for x := 0; x < 10; x++ {
			if x < 5 {
				mask.SetGray(x, y, color.Gray{Y: 0}) // Transparent background
			} else {
				mask.SetGray(x, y, color.Gray{Y: 255}) // Opaque object
			}
		}
	}

	blendParallel(dst, src, mask)

	// In blendParallel implementation:
	// alpha := float64(mask.GrayAt(x, y).Y) / 255.0
	// rOut := uint8(alpha*float64(rv>>8) + (1-alpha)*255)
	// Where rv>>8 for 255 R is 255.
	// If alpha = 0 (background): rOut = 0 + (1)*255 = 255 (White)
	// If alpha = 1 (object): rOut = 1*255 + 0 = 255 (Red - but wait, source is Red)

	// Let's check a pixel that should be "object" (Red)
	r, g, b, _ := dst.At(7, 5).RGBA()
	if uint8(r>>8) != 255 || uint8(g>>8) != 0 || uint8(b>>8) != 0 {
		t.Errorf("Expected red pixel at (7,5), got R:%d G:%d B:%d", r>>8, g>>8, b>>8)
	}

	// Let's check a pixel that should be "background" (White)
	r, g, b, _ = dst.At(2, 5).RGBA()
	if uint8(r>>8) != 255 || uint8(g>>8) != 255 || uint8(b>>8) != 255 {
		t.Errorf("Expected white pixel at (2,5), got R:%d G:%d B:%d", r>>8, g>>8, b>>8)
	}
}

func TestResizeGrayBlur5O(t *testing.T) {
	r := &RemBG{
		blurPool: newBlurBufferPool(),
	}

	src := image.NewGray(image.Rect(0, 0, 10, 10))
	// Draw a white dot in the middle
	src.SetGray(5, 5, color.Gray{Y: 255})

	// Resize to 20x20
	dst := r.resizeGrayBlur5O(src, 20, 20)

	if dst.Bounds().Dx() != 20 || dst.Bounds().Dy() != 20 {
		t.Errorf("Expected 20x20 bounds, got %v", dst.Bounds())
	}

	// Check if the blur spread the values
	foundGray := false
	for y := 0; y < 20; y++ {
		for x := 0; x < 20; x++ {
			val := dst.GrayAt(x, y).Y
			if val > 0 && val < 255 {
				foundGray = true
				break
			}
		}
	}

	if !foundGray {
		t.Error("Expected blur to create intermediate gray values")
	}
}
