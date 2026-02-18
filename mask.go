package rmbg

import (
	"image"
	"image/color"

	"github.com/disintegration/imaging"
)

type Mask func(img image.Image) *image.Gray

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
	dy, dx := bounds.Dy(), bounds.Dx()

	stepY := dy / 5
	stepX := dx / 5
	if stepY < 1 {
		stepY = 1
	}
	if stepX < 1 {
		stepX = 1
	}

	for y := bounds.Min.Y; y < bounds.Max.Y; y += stepY {
		for x := bounds.Min.X; x < bounds.Max.X; x += stepX {
			_, _, _, a := img.At(x, y).RGBA()
			if a < 0xffff {
				return true
			}
		}
	}
	return false
}

func MaskFromAlpha(img image.Image) *image.Gray {
	bounds := img.Bounds()

	switch src := img.(type) {
	case *image.RGBA:
		return maskFromImage(src.Pix, src.Stride, bounds)
	case *image.NRGBA:
		return maskFromImage(src.Pix, src.Stride, bounds)
	case *image.Gray:
		mask := image.NewGray(bounds)
		for i := range mask.Pix {
			mask.Pix[i] = 255
		}
		return mask
	}

	mask := image.NewGray(bounds)
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			_, _, _, a := img.At(x, y).RGBA()
			mask.SetGray(x, y, color.Gray{Y: uint8(a >> 8)})
		}
	}
	return mask
}

func maskFromImage(srcPix []uint8, srcStride int, bounds image.Rectangle) *image.Gray {
	w, h := bounds.Dx(), bounds.Dy()
	mask := image.NewGray(bounds)

	dstPix := mask.Pix
	dstStride := mask.Stride

	for y := range h {
		srcLine := srcPix[y*srcStride : y*srcStride+w*4]
		dstLine := dstPix[y*dstStride : y*dstStride+w]

		for x := range w {
			dstLine[x] = srcLine[x*4+3]
		}
	}
	return mask
}

func MaskFromBackground(img image.Image, bg color.Color, tolerance float64) *image.Gray {
	bounds := img.Bounds()
	mask := image.NewGray(bounds)

	bgR, bgG, bgB, _ := bg.RGBA()
	toleranceSq := tolerance * tolerance * 257.0 * 257.0

	switch src := img.(type) {
	case *image.RGBA:
		return maskFromBackground(src.Pix, src.Stride, bounds, int64(bgR), int64(bgG), int64(bgB), toleranceSq)
	case *image.NRGBA:
		return maskFromBackground(src.Pix, src.Stride, bounds, int64(bgR), int64(bgG), int64(bgB), toleranceSq)
	}

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := img.At(x, y).RGBA()

			dr := int64(r) - int64(bgR)
			dg := int64(g) - int64(bgG)
			db := int64(b) - int64(bgB)
			distSq := float64(dr*dr + dg*dg + db*db)

			if distSq > toleranceSq {
				mask.SetGray(x, y, color.Gray{Y: 255})
			}
		}
	}
	return mask
}

func maskFromBackground(srcPix []uint8, srcStride int, bounds image.Rectangle, bgR, bgG, bgB int64, toleranceSq float64) *image.Gray {
	w, h := bounds.Dx(), bounds.Dy()
	mask := image.NewGray(bounds)

	dstPix := mask.Pix
	dstStride := mask.Stride

	for y := range h {
		srcLine := srcPix[y*srcStride : y*srcStride+w*4]
		dstLine := dstPix[y*dstStride : y*dstStride+w]

		for x := range w {
			i := x * 4
			r := int64(srcLine[i]) << 8
			g := int64(srcLine[i+1]) << 8
			b := int64(srcLine[i+2]) << 8

			dr := r - bgR
			dg := g - bgG
			db := b - bgB
			distSq := float64(dr*dr + dg*dg + db*db)

			if distSq > toleranceSq {
				dstLine[x] = 255
			}
		}
	}
	return mask
}

func MaskFromEdges(img image.Image, threshold float64) *image.Gray {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()

	gray := convertToGrayscale(img)
	thresholdSq := threshold * threshold
	mask := image.NewGray(bounds)

	grayPix := gray.Pix
	maskPix := mask.Pix
	stride := gray.Stride

	for y := 1; y < h-1; y++ {
		for x := 1; x < w-1; x++ {
			idx := y*stride + x

			p0 := float64(grayPix[idx-stride-1]) // top-left
			p1 := float64(grayPix[idx-stride])   // top
			p2 := float64(grayPix[idx-stride+1]) // top-right
			p3 := float64(grayPix[idx-1])        // left
			p5 := float64(grayPix[idx+1])        // right
			p6 := float64(grayPix[idx+stride-1]) // bottom-left
			p7 := float64(grayPix[idx+stride])   // bottom
			p8 := float64(grayPix[idx+stride+1]) // bottom-right

			// Sobel X: [-1 0 1; -2 0 2; -1 0 1]
			sumX := -p0 + p2 - 2*p3 + 2*p5 - p6 + p8

			// Sobel Y: [-1 -2 -1; 0 0 0; 1 2 1]
			sumY := -p0 - 2*p1 - p2 + p6 + 2*p7 + p8

			magSq := sumX*sumX + sumY*sumY
			if magSq > thresholdSq {
				maskPix[idx] = 255
			}
		}
	}
	return mask
}

func convertToGrayscale(img image.Image) *image.Gray {
	bounds := img.Bounds()

	if g, ok := img.(*image.Gray); ok {
		gray := image.NewGray(bounds)
		copy(gray.Pix, g.Pix)
		return gray
	}

	gray := image.NewGray(bounds)
	w, h := bounds.Dx(), bounds.Dy()

	switch src := img.(type) {
	case *image.RGBA:
		return convertToGray(src.Pix, src.Stride, bounds)
	case *image.NRGBA:
		return convertToGray(src.Pix, src.Stride, bounds)
	}

	const (
		rWeight = 299
		gWeight = 587
		bWeight = 114
	)

	for y := range h {
		for x := range w {
			r, g, b, _ := img.At(bounds.Min.X+x, bounds.Min.Y+y).RGBA()
			r8, g8, b8 := r>>8, g>>8, b>>8
			yVal := (rWeight*r8 + gWeight*g8 + bWeight*b8) / 1000
			gray.SetGray(bounds.Min.X+x, bounds.Min.Y+y, color.Gray{Y: uint8(yVal)})
		}
	}
	return gray
}

func convertToGray(srcPix []uint8, srcStride int, bounds image.Rectangle) *image.Gray {
	w, h := bounds.Dx(), bounds.Dy()
	gray := image.NewGray(bounds)

	dstPix := gray.Pix
	dstStride := gray.Stride

	for y := range h {
		srcLine := srcPix[y*srcStride : y*srcStride+w*4]
		dstLine := dstPix[y*dstStride : y*dstStride+w]

		for x := range w {
			i := x * 4
			r := uint32(srcLine[i])
			g := uint32(srcLine[i+1])
			b := uint32(srcLine[i+2])
			dstLine[x] = uint8((299*r + 587*g + 114*b) / 1000)
		}
	}
	return gray
}

func detectUniformBackground(img image.Image) (color.Color, bool) {
	bounds := img.Bounds()

	samples := []image.Point{
		{bounds.Min.X, bounds.Min.Y},
		{bounds.Max.X - 1, bounds.Min.Y},
		{bounds.Min.X, bounds.Max.Y - 1},
		{bounds.Max.X - 1, bounds.Max.Y - 1},
		{bounds.Min.X + bounds.Dx()/2, bounds.Min.Y},
		{bounds.Max.X - 1, bounds.Min.Y + bounds.Dy()/2},
	}

	var rSum, gSum, bSum int64
	pixels := make([][3]int64, 0, len(samples))

	for _, pt := range samples {
		r, g, b, _ := img.At(pt.X, pt.Y).RGBA()
		rSum += int64(r)
		gSum += int64(g)
		bSum += int64(b)
		pixels = append(pixels, [3]int64{int64(r), int64(g), int64(b)})
	}

	n := int64(len(samples))
	rAvg := rSum / n
	gAvg := gSum / n
	bAvg := bSum / n

	var variance int64
	for _, p := range pixels {
		dr := p[0] - rAvg
		dg := p[1] - gAvg
		db := p[2] - bAvg
		variance += (dr*dr + dg*dg + db*db)
	}

	isUniform := variance < 2e8*n

	bg := color.RGBA{
		uint8(rAvg / 257),
		uint8(gAvg / 257),
		uint8(bAvg / 257),
		255,
	}
	return bg, isUniform
}
