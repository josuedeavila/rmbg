package rmbg

import (
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

type tensorPool struct {
	inputPool  sync.Pool
	outputPool sync.Pool
}

func newTensorPool() *tensorPool {
	return &tensorPool{
		inputPool: sync.Pool{
			New: func() any {
				t, _ := ort.NewEmptyTensor[float32](ort.NewShape(1, 3, inputSize, inputSize))
				return t
			},
		},
		outputPool: sync.Pool{
			New: func() any {
				t, _ := ort.NewEmptyTensor[float32](ort.NewShape(1, 1, inputSize, inputSize))
				return t
			},
		},
	}
}

func (p *tensorPool) getInput() *ort.Tensor[float32] {
	return p.inputPool.Get().(*ort.Tensor[float32])
}

func (p *tensorPool) putInput(t *ort.Tensor[float32]) {
	p.inputPool.Put(t)
}

func (p *tensorPool) getOutput() *ort.Tensor[float32] {
	return p.outputPool.Get().(*ort.Tensor[float32])
}

func (p *tensorPool) putOutput(t *ort.Tensor[float32]) {
	p.outputPool.Put(t)
}
