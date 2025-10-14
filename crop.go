package rmbg

import (
	"fmt"
	"image"
	"image/color"
	"math"

	"github.com/disintegration/imaging"
	ort "github.com/yalue/onnxruntime_go"
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
			Margin:       20,
			MinThreshold: 10,
		}
	}

	inputTensor := r.tensorPool.getInput()
	outputTensor := r.tensorPool.getOutput()
	defer func() {
		r.tensorPool.putInput(inputTensor)
		r.tensorPool.putOutput(outputTensor)
	}()

	resized := imaging.Resize(img, inputSize, inputSize, imaging.Linear)
	nrgba := imaging.Clone(resized)
	pix := nrgba.Pix
	stride := nrgba.Stride

	inputData := inputTensor.GetData()
	for y := range inputSize {
		row := pix[y*stride : y*stride+inputSize*4]
		for x := range inputSize {
			base := x * 4
			r := (float32(row[base+0])/255.0 - mean[0]) / std[0]
			g := (float32(row[base+1])/255.0 - mean[1]) / std[1]
			b := (float32(row[base+2])/255.0 - mean[2]) / std[2]
			inputData[(0*inputSize+y)*inputSize+x] = r
			inputData[(1*inputSize+y)*inputSize+x] = g
			inputData[(2*inputSize+y)*inputSize+x] = b
		}
	}

	err := r.RunInference([]ort.Value{inputTensor}, []ort.Value{outputTensor})
	if err != nil {
		return nil, fmt.Errorf("inference failed: %w", err)
	}

	data := outputTensor.GetData()
	maskImg := image.NewGray(image.Rect(0, 0, inputSize, inputSize))
	threshold := otsuThreshold(data)

	for i, v := range data {
		s := 1.0 / (1.0 + float32(math.Exp(float64(-v))))
		val := uint8(0)
		if s > threshold {
			val = 255
		}
		maskImg.SetGray(i%inputSize, i/inputSize, color.Gray{Y: val})
	}

	return crop(img, maskImg, config)
}

func detectObjectBounds(mask *image.Gray, minThreshold uint8) *objectBounds {
	bounds := mask.Bounds()
	minX, minY := bounds.Max.X, bounds.Max.Y
	maxX, maxY := 0, 0
	foundPixel := false

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			if mask.GrayAt(x, y).Y >= minThreshold {
				foundPixel = true
				if x < minX {
					minX = x
				}
				if x > maxX {
					maxX = x
				}
				if y < minY {
					minY = y
				}
				if y > maxY {
					maxY = y
				}
			}
		}
	}

	if !foundPixel {
		return nil
	}

	return &objectBounds{
		MinX:    minX,
		MinY:    minY,
		MaxX:    maxX,
		MaxY:    maxY,
		Width:   maxX - minX,
		Height:  maxY - minY,
		CenterX: minX + (maxX-minX)/2,
		CenterY: minY + (maxY-minY)/2,
	}
}

// SmartCropFromMask performs a smart crop using an existing mask
func (engine *RemBG) SmartCropFromMask(img image.Image, maskFunc Mask, config *CropConfig) (image.Image, error) {
	if config == nil {
		config = &CropConfig{
			Margin:       20,
			MinThreshold: 10,
		}
	}

	return crop(img, maskFunc(img), config)
}

func crop(img image.Image, maskImg *image.Gray, config *CropConfig) (image.Image, error) {
	bounds := img.Bounds()
	origW, origH := bounds.Dx(), bounds.Dy()
	objBounds := detectObjectBounds(maskImg, config.MinThreshold)
	if objBounds == nil {
		return nil, fmt.Errorf("no object detected in image")
	}

	scaledBounds := objBounds // mask and img are same size

	margin := config.Margin
	if config.MarginPercent > 0 {
		marginX := int(float64(scaledBounds.Width) * config.MarginPercent)
		marginY := int(float64(scaledBounds.Height) * config.MarginPercent)
		margin = max(marginX, marginY)
	}

	cropMinX := max(0, scaledBounds.MinX-margin)
	cropMinY := max(0, scaledBounds.MinY-margin)
	cropMaxX := min(origW, scaledBounds.MaxX+margin)
	cropMaxY := min(origH, scaledBounds.MaxY+margin)

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
