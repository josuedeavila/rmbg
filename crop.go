package rmbg

import (
	"fmt"
	"image"

	"github.com/disintegration/imaging"
)

// CropConfig configures the behavior of the smart crop
type CropConfig struct {
	// Margin is the margin in pixels around the detected object (default: 20)
	Margin int
	// MarginPercent is the margin as a percentage of the object dimensions (overrides Margin if > 0)
	MarginPercent float64
	// MinThreshold is the minimum value of the mask to consider as part of the object (0-255, default: 10)
	MinThreshold uint8
	// SquareCrop forces the crop to be square, using the largest dimension
	SquareCrop bool
}

type objectBounds struct {
	MinX, MinY, MaxX, MaxY int
	Width, Height          int
	CenterX, CenterY       int
}

// SmartCrop removes the background and performs a smart crop focusing on the object
func (r *RemBG) SmartCrop(img image.Image, config *CropConfig) (image.Image, error) {
	if config == nil {
		config = &CropConfig{
			Margin:       10,
			MinThreshold: 10,
		}
	}

	maskImg, err := r.predictMask(img)
	if err != nil {
		return nil, err
	}
	bounds := img.Bounds()
	origW, origH := bounds.Dx(), bounds.Dy()
	return crop(img, maskImg, config,
		float64(origW)/float64(inputSize),
		float64(origH)/float64(inputSize))
}

func detectObjectBounds(mask *image.Gray, minThreshold uint8) (objectBounds, bool) {
	bounds := mask.Bounds()
	minX, minY := bounds.Max.X, bounds.Max.Y
	maxX, maxY := 0, 0
	foundPixel := false

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			if mask.GrayAt(x, y).Y >= minThreshold {
				foundPixel = true
				minX = min(minX, x)
				maxX = max(maxX, x)
				minY = min(minY, y)
				maxY = max(maxY, y)
			}
		}
	}

	if !foundPixel {
		return objectBounds{}, false
	}

	return objectBounds{
		MinX:    minX,
		MinY:    minY,
		MaxX:    maxX,
		MaxY:    maxY,
		Width:   maxX - minX,
		CenterX: minX + (maxX-minX)/2,
		Height:  maxY - minY,
		CenterY: minY + (maxY-minY)/2,
	}, true
}

// SmartCropFromMask performs a smart crop using an existing mask
func (engine *RemBG) SmartCropFromMask(img image.Image, maskFunc Mask, config *CropConfig) (image.Image, error) {
	if config == nil {
		config = &CropConfig{
			Margin:       20,
			MinThreshold: 10,
		}
	}

	return crop(img, maskFunc(img), config, 1.0, 1.0)
}

func crop(
	img image.Image,
	maskImg *image.Gray,
	config *CropConfig,
	scaleX, scaleY float64,
) (image.Image, error) {
	if maskImg == nil {
		return nil, fmt.Errorf("mask image is nil")
	}

	objBounds, found := detectObjectBounds(maskImg, config.MinThreshold)
	if !found {
		return nil, fmt.Errorf("no object detected in image")
	}

	bounds := img.Bounds()
	origW, origH := bounds.Dx(), bounds.Dy()

	// Scale from mask space to original space
	scaled := &objectBounds{
		MinX: int(float64(objBounds.MinX) * scaleX),
		MinY: int(float64(objBounds.MinY) * scaleY),
		MaxX: int(float64(objBounds.MaxX) * scaleX),
		MaxY: int(float64(objBounds.MaxY) * scaleY),
	}
	scaled.Width = scaled.MaxX - scaled.MinX
	scaled.Height = scaled.MaxY - scaled.MinY

	// Calculate margin
	margin := config.Margin
	if config.MarginPercent > 0 {
		marginX := int(float64(scaled.Width) * config.MarginPercent)
		marginY := int(float64(scaled.Height) * config.MarginPercent)
		margin = max(margin, max(marginX, marginY))
	}

	cropMinX := max(0, scaled.MinX-margin)
	cropMinY := max(0, scaled.MinY-margin)
	cropMaxX := min(origW, scaled.MaxX+margin)
	cropMaxY := min(origH, scaled.MaxY+margin)

	// Make square if requested
	if config.SquareCrop {
		cropW := cropMaxX - cropMinX
		cropH := cropMaxY - cropMinY
		if cropW > cropH {
			diff := cropW - cropH
			cropMinY = max(0, cropMinY-diff/2)
			cropMaxY = min(origH, cropMaxY+diff/2)
		} else if cropH > cropW {
			diff := cropH - cropW
			cropMinX = max(0, cropMinX-diff/2)
			cropMaxX = min(origW, cropMaxX+diff/2)
		}
	}

	rect := image.Rect(cropMinX, cropMinY, cropMaxX, cropMaxY)
	return imaging.Crop(img, rect), nil
}
