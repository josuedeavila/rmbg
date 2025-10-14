package rmbg

import (
	"image"
	"image/color"
	"math"

	"github.com/disintegration/imaging"
)

type Mask func(img image.Image) *image.Gray

// AutoMask automatically chooses the best method to generate a mask:
// 1. Use alpha channel if available.
// 2. Detect background color if mostly uniform.
// 3. Fallback to edge-based Sobel mask.
func AutoMask(img image.Image) *image.Gray {
	if hasAlpha(img) {
		return MaskFromAlpha(img)
	}

	bgColor, uniform := detectUniformBackground(img)
	if uniform {
		return MaskFromBackground(img, bgColor, 200)
	}

	blurred := imaging.Blur(img, 1.0)
	return MaskFromEdges(blurred, 200)
}

func hasAlpha(img image.Image) bool {
	bounds := img.Bounds()
	for y := bounds.Min.Y; y < bounds.Max.Y; y += bounds.Dy() / 10 {
		for x := bounds.Min.X; x < bounds.Max.X; x += bounds.Dx() / 10 {
			_, _, _, a := img.At(x, y).RGBA()
			if a < 0xffff {
				return true
			}
		}
	}
	return false
}

// MaskFromAlpha uses the image’s alpha channel as mask.
func MaskFromAlpha(img image.Image) *image.Gray {
	bounds := img.Bounds()
	mask := image.NewGray(bounds)
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			_, _, _, a := img.At(x, y).RGBA()
			mask.SetGray(x, y, color.Gray{Y: uint8(a >> 8)})
		}
	}
	return mask
}

// MaskFromBackground builds a mask by comparing each pixel to a given background color.
func MaskFromBackground(img image.Image, bg color.Color, tolerance float64) *image.Gray {
	bounds := img.Bounds()
	mask := image.NewGray(bounds)

	bgR, bgG, bgB, _ := bg.RGBA()
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			dist := math.Sqrt(float64((r-bgR)*(r-bgR)+(g-bgG)*(g-bgG)+(b-bgB)*(b-bgB))) / 257.0
			if dist > tolerance {
				mask.SetGray(x, y, color.Gray{Y: 255})
			} else {
				mask.SetGray(x, y, color.Gray{Y: 0})
			}
		}
	}
	return mask
}

// MaskFromEdges detects edges using Sobel operator (pure Go).
func MaskFromEdges(img image.Image, threshold float64) *image.Gray {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()

	// Convert to grayscale
	gray := image.NewGray(bounds)
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			yVal := 0.299*float64(r>>8) + 0.587*float64(g>>8) + 0.114*float64(b>>8)
			gray.SetGray(x, y, color.Gray{Y: uint8(yVal)})
		}
	}

	gx := [3][3]float64{{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}}
	gy := [3][3]float64{{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}}
	mask := image.NewGray(bounds)

	for y := 1; y < h-1; y++ {
		for x := 1; x < w-1; x++ {
			var sumX, sumY float64
			for ky := -1; ky <= 1; ky++ {
				for kx := -1; kx <= 1; kx++ {
					p := float64(gray.GrayAt(x+kx, y+ky).Y)
					sumX += gx[ky+1][kx+1] * p
					sumY += gy[ky+1][kx+1] * p
				}
			}
			mag := math.Sqrt(sumX*sumX + sumY*sumY)
			if mag > threshold {
				mask.SetGray(x, y, color.Gray{Y: 255})
			} else {
				mask.SetGray(x, y, color.Gray{Y: 0})
			}
		}
	}
	return mask
}

func detectUniformBackground(img image.Image) (color.Color, bool) {
	bounds := img.Bounds()
	sample := []color.Color{
		img.At(bounds.Min.X, bounds.Min.Y),
		img.At(bounds.Max.X-1, bounds.Min.Y),
		img.At(bounds.Min.X, bounds.Max.Y-1),
		img.At(bounds.Max.X-1, bounds.Max.Y-1),
	}

	var rSum, gSum, bSum float64
	var pixels [][3]float64

	for _, c := range sample {
		r, g, b, _ := c.RGBA()
		rSum += float64(r)
		gSum += float64(g)
		bSum += float64(b)
		pixels = append(pixels, [3]float64{float64(r), float64(g), float64(b)})
	}

	rAvg := rSum / 4
	gAvg := gSum / 4
	bAvg := bSum / 4

	var variance float64
	for _, p := range pixels {
		dr := p[0] - rAvg
		dg := p[1] - gAvg
		db := p[2] - bAvg
		variance += (dr*dr + dg*dg + db*db)
	}
	variance /= 4

	// If variance is small, background is likely uniform
	isUniform := variance < 2e8
	bg := color.RGBA{uint8(rAvg / 257), uint8(gAvg / 257), uint8(bAvg / 257), 255}
	return bg, isUniform
}
