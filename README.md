# rmbg

🎨 **High-performance background removal library for Go**

A fast and efficient Go library for removing backgrounds from images using ONNX Runtime and the U²-Net model. Perfect for e-commerce, profile pictures, product photography, and image preprocessing pipelines.

## ✨ Features

- 🎯 **Smart Cropping**: Automatically detect and crop around the main subject
- 🖼️ **Multiple Mask Strategies**: Auto-detect, alpha channel, edge detection, or background color detection
- 🔧 **Configurable**: Fine-tune thread count, memory usage, and processing parameters
- 🔄 **Reusable Sessions**: Session pooling for optimal performance in concurrent environments

## 📋 Requirements

- Go 1.25.0 or higher
- ONNX Runtime library installed on your system

## 🚀 Installation

```bash
go get github.com/josuedeavila/rmbg
```

## 📥 Model Download

Download the U²-Net ONNX model:

```bash
mkdir -p models
cd models
# Download u2netp model (smaller, faster)
wget https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx
```

Or use the full U²-Net model for better quality:

```bash
wget https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx
```

## 💡 Quick Start

### Basic Background Removal

```go
package main

import (
    "fmt"
    "github.com/disintegration/imaging"
    "github.com/josuedeavila/rmbg"
)

func main() {
    // Initialize the engine
    cfg := &rmbg.Config{
        ModelPath:         "./models/u2netp.onnx",
        IntraOpNumThreads: 4,
        InterOpNumThreads: 2,
        CpuMemArena:       false,
        MemPattern:        true,
    }

    engine, err := rmbg.New(cfg)
    if err != nil {
        panic(err)
    }
    defer engine.Close()

    // Load image
    img, err := imaging.Open("input.jpg")
    if err != nil {
        panic(err)
    }

    // Remove background
    result, err := engine.RemoveBackground(img)
    if err != nil {
        panic(err)
    }

    // Save result
    err = imaging.Save(result, "output.png")
    if err != nil {
        panic(err)
    }

    fmt.Println("Background removed successfully!")
}
```

### Smart Crop

Automatically crop the image around the detected subject:

```go
// Remove background and crop
cropped, err := engine.SmartCrop(img, &rmbg.CropConfig{
    Margin:        20,          // Pixels around the object
    MarginPercent: 0.1,         // Or 10% of object dimensions
    MinThreshold:  10,          // Mask sensitivity (0-255)
    SquareCrop:    true,        // Force square crop
})
if err != nil {
    panic(err)
}

imaging.Save(cropped, "cropped.png")
```

### Using Custom Masks

```go
// Use existing alpha channel
result, err := engine.SmartCropFromMask(img, rmbg.MaskFromAlpha, &rmbg.CropConfig{
    Margin:       10,
    SquareCrop:   true,
    MinThreshold: 100,
})

// Auto-detect best mask strategy
result, err := engine.SmartCropFromMask(img, rmbg.AutoMask, config)

// Detect edges
result, err := engine.SmartCropFromMask(img, func(img image.Image) *image.Gray {
    return rmbg.MaskFromEdges(img, 200)
}, config)

// Detect background color
result, err := engine.SmartCropFromMask(img, func(img image.Image) *image.Gray {
    bgColor := color.RGBA{R: 255, G: 255, B: 255, A: 255} // white
    return rmbg.MaskFromBackground(img, bgColor, 50)
}, config)
```

## ⚙️ Configuration

### Engine Config

```go
type Config struct {
    // Path to ONNX model file
    ModelPath string

    // Number of threads for intra-op parallelism (default: 1)
    // Higher values = more parallel ops within layers
    IntraOpNumThreads int

    // Number of threads for inter-op parallelism (default: 1)
    // Higher values = more parallel execution between layers
    InterOpNumThreads int

    // Enable CPU memory arena (default: false)
    // Can improve performance but uses more memory
    CpuMemArena bool

    // Enable memory pattern optimization (default: true)
    MemPattern bool
}
```

### Crop Config

```go
type CropConfig struct {
    // Fixed margin in pixels around detected object
    Margin int

    // Margin as percentage of object dimensions (overrides Margin)
    MarginPercent float64

    // Minimum mask value to consider as object (0-255)
    MinThreshold uint8

    // Force square crop using largest dimension
    SquareCrop bool
}
```

## 🎯 Use Cases

- **E-commerce**: Product photography with clean backgrounds
- **Profile Pictures**: Automatic background removal for avatars
- **Image Processing Pipelines**: Batch processing of images
- **AR/VR Applications**: Subject extraction for compositing
- **Document Scanning**: Remove backgrounds from scanned objects
- **Creative Tools**: Photo editing and manipulation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
