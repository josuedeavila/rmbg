package rmbg

import (
	"testing"
)

func TestBlurBufferPool(t *testing.T) {
	pool := newBlurBufferPool()

	t.Run("GetAndPut", func(t *testing.T) {
		size := 1024
		buf := pool.get(size)
		if len(buf.tmp) != size || len(buf.hPass) != size {
			t.Errorf("Expected size %d, got %d and %d", size, len(buf.tmp), len(buf.hPass))
		}
		pool.put(buf)
	})

	t.Run("Resizing", func(t *testing.T) {
		size1 := 100
		buf1 := pool.get(size1)
		cap1 := cap(buf1.tmp)
		pool.put(buf1)

		// Test growing
		size2 := 200
		buf2 := pool.get(size2)
		if len(buf2.tmp) != size2 {
			t.Errorf("Expected size %d, got %d", size2, len(buf2.tmp))
		}
		pool.put(buf2)

		// Test shrinking (capacity should be retained)
		size3 := 50
		buf3 := pool.get(size3)
		if len(buf3.tmp) != size3 {
			t.Errorf("Expected size %d, got %d", size3, len(buf3.tmp))
		}
		if cap(buf3.tmp) < cap1 {
			t.Errorf("Expected capacity to be at least %d, got %d", cap1, cap(buf3.tmp))
		}
		pool.put(buf3)
	})
}

func TestTensorPool(t *testing.T) {
	// tensorPool relies on ONNX Runtime environment.
	// If it's not initialized (e.g. missing shared libraries),
	// the New functions might return nil or the init() might have panicked.

	pool := newTensorPool()

	t.Run("InputTensor", func(t *testing.T) {
		input := pool.getInput()
		if input == nil {
			t.Log("Input tensor is nil - ORT environment might not be fully initialized in this environment")
			return
		}
		defer pool.putInput(input)

		shape := input.GetShape()
		expected := []int64{1, 3, inputSize, inputSize}
		if len(shape) != len(expected) {
			t.Errorf("Expected shape length %d, got %d", len(expected), len(shape))
			return
		}
		for i := range expected {
			if shape[i] != expected[i] {
				t.Errorf("Expected shape[%d] = %d, got %d", i, expected[i], shape[i])
			}
		}
	})

	t.Run("OutputTensor", func(t *testing.T) {
		output := pool.getOutput()
		if output == nil {
			t.Log("Output tensor is nil - ORT environment might not be fully initialized in this environment")
			return
		}
		defer pool.putOutput(output)

		shape := output.GetShape()
		expected := []int64{1, 1, inputSize, inputSize}
		if len(shape) != len(expected) {
			t.Errorf("Expected shape length %d, got %d", len(expected), len(shape))
			return
		}
		for i := range expected {
			if shape[i] != expected[i] {
				t.Errorf("Expected shape[%d] = %d, got %d", i, expected[i], shape[i])
			}
		}
	})
}
