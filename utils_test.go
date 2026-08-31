package rmbg

import (
	"image"
	"image/color"
	"testing"
)

func TestClamp(t *testing.T) {
	tests := []struct {
		v, min, max int
		want        int
	}{
		{5, 0, 10, 5},
		{-1, 0, 10, 0},
		{11, 0, 10, 10},
		{0, 0, 10, 0},
		{10, 0, 10, 10},
	}

	for _, tt := range tests {
		if got := clamp(tt.v, tt.min, tt.max); got != tt.want {
			t.Errorf("clamp(%d, %d, %d) = %d; want %d", tt.v, tt.min, tt.max, got, tt.want)
		}
	}
}

func TestDetectUniformBackground(t *testing.T) {
	t.Run("Uniform", func(t *testing.T) {
		img := image.NewRGBA(image.Rect(0, 0, 100, 100))
		blue := color.RGBA{0, 0, 255, 255}
		for y := 0; y < 100; y++ {
			for x := 0; x < 100; x++ {
				img.Set(x, y, blue)
			}
		}

		bg, uniform := detectUniformBackground(img)
		if !uniform {
			t.Errorf("expected uniform background")
		}
		r, g, b, _ := bg.RGBA()
		if uint8(r>>8) != 0 || uint8(g>>8) != 0 || uint8(b>>8) != 255 {
			t.Errorf("detected wrong color: %v", bg)
		}
	})

	t.Run("NonUniform", func(t *testing.T) {
		img := image.NewRGBA(image.Rect(0, 0, 100, 100))
		for y := 0; y < 100; y++ {
			for x := 0; x < 100; x++ {
				if x < 50 {
					img.Set(x, y, color.White)
				} else {
					img.Set(x, y, color.Black)
				}
			}
		}

		_, uniform := detectUniformBackground(img)
		if uniform {
			t.Errorf("expected non-uniform background")
		}
	})
}

func TestHasAlpha(t *testing.T) {
	t.Run("HasAlpha", func(t *testing.T) {
		img := image.NewRGBA(image.Rect(0, 0, 10, 10))
		img.Set(5, 5, color.RGBA{255, 0, 0, 128})
		if !hasAlpha(img) {
			t.Errorf("expected alpha channel detection")
		}
	})

	t.Run("Opaque", func(t *testing.T) {
		img := image.NewRGBA(image.Rect(0, 0, 10, 10))
		for y := 0; y < 10; y++ {
			for x := 0; x < 10; x++ {
				img.Set(x, y, color.RGBA{255, 255, 255, 255})
			}
		}
		if hasAlpha(img) {
			t.Errorf("expected no alpha channel detection")
		}
	})
}
