package rmbg

import (
	"image"
	"image/color"
	"testing"
)

func TestMaskFromAlpha(t *testing.T) {
	bounds := image.Rect(0, 0, 10, 10)

	t.Run("RGBA", func(t *testing.T) {
		img := image.NewRGBA(bounds)
		for y := 0; y < 10; y++ {
			for x := 0; x < 10; x++ {
				img.SetRGBA(x, y, color.RGBA{0, 0, 0, uint8(x * 25)})
			}
		}
		mask := MaskFromAlpha(img)
		if mask.Bounds() != bounds {
			t.Errorf("expected bounds %v, got %v", bounds, mask.Bounds())
		}
		for x := 0; x < 10; x++ {
			val := mask.GrayAt(x, 0).Y
			if val != uint8(x*25) {
				t.Errorf("at x=%d, expected mask value %d, got %d", x, x*25, val)
			}
		}
	})

	t.Run("NRGBA", func(t *testing.T) {
		img := image.NewNRGBA(bounds)
		for y := 0; y < 10; y++ {
			for x := 0; x < 10; x++ {
				img.SetNRGBA(x, y, color.NRGBA{0, 0, 0, uint8(x * 25)})
			}
		}
		mask := MaskFromAlpha(img)
		for x := 0; x < 10; x++ {
			val := mask.GrayAt(x, 0).Y
			if val != uint8(x*25) {
				t.Errorf("at x=%d, expected mask value %d, got %d", x, x*25, val)
			}
		}
	})

	t.Run("Gray", func(t *testing.T) {
		img := image.NewGray(bounds)
		mask := MaskFromAlpha(img)
		for i := range mask.Pix {
			if mask.Pix[i] != 255 {
				t.Errorf("expected 255 for Gray image alpha, got %d", mask.Pix[i])
				break
			}
		}
	})
}

func TestMaskFromBackground(t *testing.T) {
	bounds := image.Rect(0, 0, 10, 10)
	img := image.NewRGBA(bounds)

	// Fill with white
	for i := 0; i < len(img.Pix); i++ {
		img.Pix[i] = 255
	}
	// Add a black square in the middle
	for y := 4; y < 6; y++ {
		for x := 4; x < 6; x++ {
			img.Set(x, y, color.Black)
		}
	}

	mask := MaskFromBackground(img, color.White, 10.0)

	// Background (white) should be transparent (0) in mask
	if mask.GrayAt(0, 0).Y != 0 {
		t.Errorf("expected background to be 0, got %d", mask.GrayAt(0, 0).Y)
	}
	// Object (black) should be opaque (255) in mask
	if mask.GrayAt(5, 5).Y != 255 {
		t.Errorf("expected object to be 255, got %d", mask.GrayAt(5, 5).Y)
	}
}

func TestConvertToGrayscale(t *testing.T) {
	bounds := image.Rect(0, 0, 2, 2)
	img := image.NewRGBA(bounds)
	// Set red pixel
	img.Set(0, 0, color.RGBA{255, 0, 0, 255})
	// Set green pixel
	img.Set(1, 0, color.RGBA{0, 255, 0, 255})
	// Set blue pixel
	img.Set(0, 1, color.RGBA{0, 0, 255, 255})
	// Set white pixel
	img.Set(1, 1, color.RGBA{255, 255, 255, 255})

	gray := convertToGrayscale(img)

	// Weights: R: 0.299, G: 0.587, B: 0.114
	expected := []uint8{
		uint8((299 * 255) / 1000),
		uint8((587 * 255) / 1000),
		uint8((114 * 255) / 1000),
		255,
	}

	if gray.GrayAt(0, 0).Y != expected[0] {
		t.Errorf("Red: expected %d, got %d", expected[0], gray.GrayAt(0, 0).Y)
	}
	if gray.GrayAt(1, 0).Y != expected[1] {
		t.Errorf("Green: expected %d, got %d", expected[1], gray.GrayAt(1, 0).Y)
	}
	if gray.GrayAt(0, 1).Y != expected[2] {
		t.Errorf("Blue: expected %d, got %d", expected[2], gray.GrayAt(0, 1).Y)
	}
	if gray.GrayAt(1, 1).Y != expected[3] {
		t.Errorf("White: expected %d, got %d", expected[3], gray.GrayAt(1, 1).Y)
	}
}

func TestMaskFromEdges(t *testing.T) {
	bounds := image.Rect(0, 0, 20, 20)
	img := image.NewGray(bounds)

	// Create a sharp vertical edge
	for y := 0; y < 20; y++ {
		for x := 10; x < 20; x++ {
			img.SetGray(x, y, color.Gray{Y: 255})
		}
	}

	mask := MaskFromEdges(img, 50.0)

	// Check if edge was detected (around x=10)
	foundEdge := false
	for y := 1; y < 19; y++ {
		if mask.GrayAt(10, y).Y == 255 || mask.GrayAt(9, y).Y == 255 {
			foundEdge = true
			break
		}
	}

	if !foundEdge {
		t.Errorf("expected edge to be detected")
	}

	// Check that uniform areas are not edges
	if mask.GrayAt(2, 2).Y != 0 {
		t.Errorf("uniform area detected as edge at (2,2)")
	}
	if mask.GrayAt(18, 18).Y != 0 {
		t.Errorf("uniform area detected as edge at (18,18)")
	}
}

func TestAutoMask(t *testing.T) {
	t.Run("PreferAlpha", func(t *testing.T) {
		img := image.NewRGBA(image.Rect(0, 0, 10, 10))
		img.Set(5, 5, color.RGBA{0, 0, 0, 128}) // has alpha
		mask := AutoMask(img)
		if mask.GrayAt(5, 5).Y != 128 {
			t.Errorf("expected alpha-based mask value 128, got %d", mask.GrayAt(5, 5).Y)
		}
	})

	t.Run("PreferBackground", func(t *testing.T) {
		img := image.NewRGBA(image.Rect(0, 0, 10, 10))
		// Uniform blue background
		blue := color.RGBA{0, 0, 255, 255}
		for y := 0; y < 10; y++ {
			for x := 0; x < 10; x++ {
				img.Set(x, y, blue)
			}
		}
		// Red square
		img.Set(5, 5, color.RGBA{255, 0, 0, 255})

		mask := AutoMask(img)
		if mask.GrayAt(0, 0).Y != 0 {
			t.Errorf("expected background to be 0, got %d", mask.GrayAt(0, 0).Y)
		}
		if mask.GrayAt(5, 5).Y != 255 {
			t.Errorf("expected object to be 255, got %d", mask.GrayAt(5, 5).Y)
		}
	})
}
