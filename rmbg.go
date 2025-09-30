package bgrm

import (
	"fmt"
	"image"
	"image/color"
	_ "image/jpeg"
	"image/png"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"

	"github.com/nfnt/resize"
	ort "github.com/yalue/onnxruntime_go"
)

const (
	modelURL  = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"
	inputSize = 320
)

var (
	mean = [3]float32{0.485, 0.456, 0.406}
	std  = [3]float32{0.229, 0.224, 0.225}
)

// RemBG handles background removal operations
type RemBG struct {
	session     *ort.AdvancedSession
	modelPath   string
	inputTensor *ort.Tensor[float32]
	outputs     []*ort.Tensor[float32] // store all outputs here
}

func New(modelDir string) (*RemBG, error) {
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, fmt.Errorf("failed to initialize ONNX runtime: %w", err)
	}

	if err := os.MkdirAll(modelDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create model directory: %w", err)
	}

	modelPath := filepath.Join(modelDir, "u2netp.onnx")

	// Download model if it doesn't exist
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		fmt.Println("Downloading U2-Net model...")
		if err := downloadModel(modelURL, modelPath); err != nil {
			return nil, fmt.Errorf("failed to download model: %w", err)
		}
		fmt.Println("Model downloaded successfully")
	}

	// Input tensor
	inputTensor, err := ort.NewTensor(
		ort.NewShape(1, 3, inputSize, inputSize),
		make([]float32, 1*3*inputSize*inputSize),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %w", err)
	}

	outputNames := []string{"1959", "1960", "1961", "1962", "1963", "1964", "1965"}
	outputTensors := []*ort.Tensor[float32]{}
	for _, name := range outputNames {
		t, e := ort.NewTensor(
			ort.NewShape(1, 1, inputSize, inputSize),
			make([]float32, inputSize*inputSize),
		)
		if e != nil {
			return nil, fmt.Errorf("failed to create output tensor %s: %w", name, e)
		}
		outputTensors = append(outputTensors, t)
	}

	values := make([]ort.Value, len(outputTensors))
	for i, t := range outputTensors {
		values[i] = t
	}

	session, err := ort.NewAdvancedSession(
		modelPath,
		[]string{"input.1"},
		outputNames,
		[]ort.ArbitraryTensor{inputTensor},
		values,
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create ONNX session: %w", err)
	}

	return &RemBG{
		session:     session,
		modelPath:   modelPath,
		inputTensor: inputTensor,
		outputs:     outputTensors,
	}, nil
}

// RemoveBackground removes the background from an image and replaces it with white
func (r *RemBG) RemoveBackground(img image.Image) (image.Image, error) {
	bounds := img.Bounds()

	// Resize image to model input size
	resized := resize.Resize(inputSize, inputSize, img, resize.Lanczos3)

	// Normalize and convert to CHW format
	inputData := r.inputTensor.GetData()
	idx := 0
	for c := range 3 {
		for y := range inputSize {
			for x := range inputSize {
				rv, gv, bv, _ := resized.At(x, y).RGBA()
				var val float32
				switch c {
				case 0:
					val = (float32(rv>>8)/255.0 - mean[0]) / std[0]
				case 1:
					val = (float32(gv>>8)/255.0 - mean[1]) / std[1]
				case 2:
					val = (float32(bv>>8)/255.0 - mean[2]) / std[2]
				}
				inputData[idx] = val
				idx++
			}
		}
	}

	// Run inference
	if err := r.session.Run(); err != nil {
		return nil, fmt.Errorf("inference failed: %w", err)
	}

	data := make([]float32, inputSize*inputSize)
	for _, out := range r.outputs {
		o := out.GetData()
		for i := range data {
			data[i] += o[i]
		}
	}
	for i := range data {
		data[i] /= float32(len(r.outputs)) // average
	}

	// Build soft mask image (float alpha 0–1)
	maskImg := image.NewGray(image.Rect(0, 0, inputSize, inputSize))
	treshHold := otsuThreshold(data)
	for i, v := range data {
		s := 1.0 / (1.0 + float32(math.Exp(float64(-v)))) // sigmoid (0–1)
		if s > treshHold {
			s = 1.0
		} else {
			s = 0.0
		}
		val := uint8(s * 255)
		maskImg.SetGray(i%inputSize, i/inputSize, color.Gray{Y: val})

	}

	// Resize mask to original size
	resizedMask := resize.Resize(uint(bounds.Dx()), uint(bounds.Dy()), maskImg, resize.Lanczos3)
	resizedMask = refineMask(resizedMask.(*image.Gray))

	output := image.NewRGBA(bounds)
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			rv, gv, bv, _ := img.At(x, y).RGBA()
			alpha := float64(resizedMask.At(x, y).(color.Gray).Y) / 255.0
			rOut := uint8(alpha*float64(rv>>8) + (1-alpha)*255)
			gOut := uint8(alpha*float64(gv>>8) + (1-alpha)*255)
			bOut := uint8(alpha*float64(bv>>8) + (1-alpha)*255)

			output.SetRGBA(x, y, color.RGBA{
				R: rOut,
				G: gOut,
				B: bOut,
				A: 255,
			})
		}
	}

	return output, nil
}

// RemoveBackgroundFromFile loads an image file and removes its background
func (r *RemBG) RemoveBackgroundFromFile(inputPath, outputPath string) error {
	file, err := os.Open(inputPath)
	if err != nil {
		return fmt.Errorf("failed to open input file: %w", err)
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		return fmt.Errorf("failed to decode image: %w", err)
	}

	result, err := r.RemoveBackground(img)
	if err != nil {
		return err
	}

	outFile, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("failed to create output file: %w", err)
	}

	defer outFile.Close()

	if err := png.Encode(outFile, result); err != nil {
		return fmt.Errorf("failed to encode output image: %w", err)
	}

	return nil
}

// Close releases resources
func (r *RemBG) Close() error {
	if r.session != nil {
		return r.session.Destroy()
	}
	if r.inputTensor != nil {
		return r.inputTensor.Destroy()
	}
	if r.outputs != nil {
		for _, output := range r.outputs {
			return output.Destroy()
		}
	}
	return nil
}

func downloadModel(url, filepath string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("bad status: %s", resp.Status)
	}

	out, err := os.Create(filepath)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, resp.Body)
	return err
}

func refineMask(mask *image.Gray) *image.Gray {
	bounds := mask.Bounds()
	refined := image.NewGray(bounds)

	kernel := [5][5]int{
		{1, 4, 6, 4, 1},
		{4, 16, 24, 16, 4},
		{6, 24, 36, 24, 6},
		{4, 16, 24, 16, 4},
		{1, 4, 6, 4, 1},
	}
	kernelSum := 256

	for y := bounds.Min.Y + 2; y < bounds.Max.Y-2; y++ {
		for x := bounds.Min.X + 2; x < bounds.Max.X-2; x++ {
			sum := 0
			for ky := -2; ky <= 2; ky++ {
				for kx := -2; kx <= 2; kx++ {
					sum += int(mask.GrayAt(x+kx, y+ky).Y) * kernel[ky+2][kx+2]
				}
			}
			v := uint8(sum / kernelSum)
			refined.SetGray(x, y, color.Gray{Y: v})
		}
	}

	return refined
}

func otsuThreshold(data []float32) float32 {
	hist := make([]int, 256)
	for _, v := range data {
		// Apply sigmoid first
		s := 1.0 / (1.0 + float32(math.Exp(float64(-v))))
		val := int(s * 255.0)
		val = max(val, 0)
		val = min(val, 255)
		hist[val]++
	}

	total := len(data)
	sum := 0
	for t := range 255 {
		sum += t * hist[t]
	}

	sumB, wB, wF, varMax, threshold := 0, 0, 0, 0.0, 0
	for t := range 255 {
		wB += hist[t]
		if wB == 0 {
			continue
		}
		wF = total - wB
		if wF == 0 {
			break
		}
		sumB += t * hist[t]
		mB := float64(sumB) / float64(wB)
		mF := float64(sum-sumB) / float64(wF)
		varBetween := float64(wB) * float64(wF) * (mB - mF) * (mB - mF)
		if varBetween > varMax {
			varMax = varBetween
			threshold = t
		}
	}

	return float32(threshold) / 255.0
}
