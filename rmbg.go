package rmbg

import (
	"fmt"
	"image"
	"image/color"
	"log"
	"math"
	"runtime"
	"sync"

	"github.com/disintegration/imaging"
	ort "github.com/yalue/onnxruntime_go"
)

func init() {
	for i := range 255 {
		v := float32(i)/255.0*12.0 - 6.0
		sigmoidLUT[i] = 1.0 / (1.0 + float32(math.Exp(float64(-v))))
	}

	if err := ort.InitializeEnvironment(); err != nil {
		log.Panicf("failed to init ORT env: %v", err)
	}
}

const (
	inputSize = 320
)

var (
	sigmoidLUT [256]float32
	mean       = [3]float32{0.485, 0.456, 0.406}
	std        = [3]float32{0.229, 0.224, 0.225}
)

// RemBG with session reuse and memory pooling
type RemBG struct {
	modelPath  string
	session    *ort.DynamicAdvancedSession
	sessionMu  sync.Mutex
	tensorPool *tensorPool
	blurPool   *blurBufferPool
}

func createSession(modelPath string) (*ort.DynamicAdvancedSession, error) {
	options, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("failed to create session options: %w", err)
	}
	defer options.Destroy()

	// Configure for minimal memory usage
	options.SetIntraOpNumThreads(2)
	options.SetInterOpNumThreads(1)
	options.SetCpuMemArena(false)
	options.SetMemPattern(true)
	options.SetExecutionMode(ort.ExecutionModeSequential)
	options.SetGraphOptimizationLevel(ort.GraphOptimizationLevelEnableAll)

	session, err := ort.NewDynamicAdvancedSession(
		modelPath,
		[]string{"input.1"},
		[]string{"1959"},
		options,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create ONNX session: %w", err)
	}

	return session, nil
}

// NewRemBG initializes ONNX session with memory pooling
func NewRemBG(modelPath string) (*RemBG, error) {
	session, err := createSession(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to create ONNX session: %w", err)
	}

	return &RemBG{
		modelPath:  modelPath,
		session:    session,
		tensorPool: newTensorPool(),
		blurPool:   newBlurBufferPool(),
	}, nil
}

// Close destroys the session
func (r *RemBG) Close() error {
	if r.session != nil {
		return r.session.Destroy()
	}
	return ort.DestroyEnvironment()
}

// RemoveBackground processes image with memory pooling
func (r *RemBG) RemoveBackground(img image.Image) (image.Image, error) {
	inputTensor := r.tensorPool.getInput()
	outputTensor := r.tensorPool.getOutput()
	defer func() {
		r.tensorPool.putInput(inputTensor)
		r.tensorPool.putOutput(outputTensor)
	}()

	resized := imaging.Resize(img, inputSize, inputSize, imaging.Linear)
	nrgba := imaging.Clone(resized) // ensures it's *image.NRGBA
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
		if s > threshold {
			s = 1.0
		} else {
			s = 0.0
		}
		val := uint8(s * 255)
		maskImg.SetGray(i%inputSize, i/inputSize, color.Gray{Y: val})
	}

	bounds := img.Bounds()
	resizedMask := r.resizeGrayBlur5O(maskImg, bounds.Dx(), bounds.Dy())

	output := image.NewRGBA(bounds)
	blendParallel(output, img, resizedMask)

	return output, nil
}

func blendParallel(dst *image.RGBA, src image.Image, mask *image.Gray) {
	bounds := src.Bounds()
	numCPU := runtime.NumCPU()
	var wg sync.WaitGroup
	chunk := (bounds.Dy() + numCPU - 1) / numCPU

	for i := range runtime.NumCPU() {
		startY := i * chunk
		endY := min(startY+chunk, bounds.Dy())
		if startY >= endY {
			continue
		}

		wg.Go(func() {
			for y := startY; y < endY; y++ {
				for x := bounds.Min.X; x < bounds.Max.X; x++ {
					rv, gv, bv, _ := src.At(x, y).RGBA()
					alpha := float64(mask.GrayAt(x, y).Y) / 255.0
					rOut := uint8(alpha*float64(rv>>8) + (1-alpha)*255)
					gOut := uint8(alpha*float64(gv>>8) + (1-alpha)*255)
					bOut := uint8(alpha*float64(bv>>8) + (1-alpha)*255)
					dst.SetRGBA(x, y, color.RGBA{R: rOut, G: gOut, B: bOut, A: 255})
				}
			}
		})

		wg.Go(func() {
			for y := startY; y < endY; y++ {
				for x := bounds.Min.X; x < bounds.Max.X; x++ {
					rv, gv, bv, _ := src.At(x, y).RGBA()
					alpha := float64(mask.GrayAt(x, y).Y) / 255.0
					rOut := uint8(alpha*float64(rv>>8) + (1-alpha)*255)
					gOut := uint8(alpha*float64(gv>>8) + (1-alpha)*255)
					bOut := uint8(alpha*float64(bv>>8) + (1-alpha)*255)
					dst.SetRGBA(x, y, color.RGBA{R: rOut, G: gOut, B: bOut, A: 255})
				}
			}
		})
	}

	wg.Wait()
}

// Optimized resize with flat arrays instead of 2D slices
func (r *RemBG) resizeGrayBlur5O(src *image.Gray, newW, newH int) *image.Gray {
	srcB := src.Bounds()
	dst := image.NewGray(image.Rect(0, 0, newW, newH))

	xRatio := float64(srcB.Dx()) / float64(newW)
	yRatio := float64(srcB.Dy()) / float64(newH)

	// Get buffer from pool
	bufSize := newW * newH
	buf := r.blurPool.get(bufSize)
	defer r.blurPool.put(buf)

	tmp := buf.tmp
	hPass := buf.hPass

	for y := range newH {
		sy := yRatio * float64(y)
		y0 := int(sy)
		y1 := min(y0+1, srcB.Dy()-1)
		yLerp := sy - float64(y0)

		for x := range newW {
			sx := xRatio * float64(x)
			x0 := int(sx)
			x1 := min(x0+1, srcB.Dx()-1)
			xLerp := sx - float64(x0)

			p00 := float64(src.GrayAt(x0, y0).Y)
			p10 := float64(src.GrayAt(x1, y0).Y)
			p01 := float64(src.GrayAt(x0, y1).Y)
			p11 := float64(src.GrayAt(x1, y1).Y)

			top := p00 + (p10-p00)*xLerp
			bottom := p01 + (p11-p01)*xLerp
			tmp[y*newW+x] = uint8(top + (bottom-top)*yLerp)
		}
	}

	w, h := newW, newH
	window := 5
	radius := window / 2

	for y := range h {
		sum := 0
		rowOffset := y * w
		for k := -radius; k <= radius; k++ {
			xi := clamp(k, 0, w-1)
			sum += int(tmp[rowOffset+xi])
		}
		hPass[rowOffset] = uint8(sum / window)

		for x := 1; x < w; x++ {
			out := min(x+radius, w-1)
			in := max(x-radius-1, 0)
			sum += int(tmp[rowOffset+out]) - int(tmp[rowOffset+in])
			hPass[rowOffset+x] = uint8(sum / window)
		}
	}

	for x := range w {
		sum := 0
		for k := -radius; k <= radius; k++ {
			yi := clamp(k, 0, h-1)
			sum += int(hPass[yi*w+x])
		}
		dst.SetGray(x, 0, color.Gray{Y: uint8(sum / window)})

		for y := 1; y < h; y++ {
			out := min(y+radius, h-1)
			in := max(y-radius-1, 0)
			sum += int(hPass[out*w+x]) - int(hPass[in*w+x])
			dst.SetGray(x, y, color.Gray{Y: uint8(sum / window)})
		}
	}

	return dst
}

func (r *RemBG) RunInference(input []ort.Value, output []ort.Value) error {
	r.sessionMu.Lock()
	err := r.session.Run(input, output)
	r.sessionMu.Unlock()
	return err
}

func clamp(v, min, max int) int {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

func otsuThreshold(data []float32) float32 {
	hist := make([]int, 256)
	for _, v := range data {
		s := sigmoidLUT[int((v+6.0)/12.0*255.0)]
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
