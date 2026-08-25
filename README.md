# GPU Image Preprocessing Pipeline

A CUDA-accelerated image preprocessing pipeline following the standard **ImageNet preprocessing spec** used by models like ResNet, VGG, and EfficientNet.

Four kernels run entirely on the GPU:

```
Color Image → Grayscale → Resize (256×256) → Center Crop (224×224) → Normalize
```

---

## Before / After

| Original | Output |
|----------|--------|
| ![original](assets/original.png) | ![output](assets/output.png) |

> Color JPEG → grayscale, resized, center-cropped — processed entirely on GPU.

---

## Pipeline

| Stage | Kernel | What it does |
|---|---|---|
| 1 | `grayscaleKernel` | RGB → single float using ITU-R BT.601 luma weights |
| 2 | `resizeKernel` | Bilinear interpolation -> 256×256 |
| 3 | `cropKernel` | Center crop → 224×224 (16px margin removed per side) |
| 4 | `normalizeKernel` | `(pixel/255 - 0.485) / 0.229` - ImageNet mean/std |

Each kernel is fully parallelized: one CUDA thread per output pixel.

---

## Requirements

- NVIDIA GPU (CUDA-capable)
- CUDA Toolkit ≥ 11.0
- No other dependencies — image I/O uses [`stb_image`](https://github.com/nothings/stb) (single-header, included automatically)

---

## Run Locally

**Step 1 — Download stb headers:**
```bash
wget -q https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
wget -q https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h
```

**Step 2 — Clone and compile:**
```bash
git clone https://github.com/saathviksheerla/gpu_image_pipeline
cd gpu_image_pipeline
nvcc pipeline.cu -o gpu_pipeline
```

**Step 3 — Run on any image:**
```bash
./gpu_pipeline sample_inputs/dog.jpg
./gpu_pipeline sample_inputs/cat.jpg
./gpu_pipeline sample_inputs/octopus.jpg   # parallel computing vibes
./gpu_pipeline your_image.jpg
```

Outputs saved in the current directory:
- `original.png` — re-saved input (color)
- `output.png` — grayscale, resized, cropped

---

## Run on Google Colab

No local GPU? Use Colab (free T4 GPU):

**Step 1 — Download stb headers:**
```bash
!wget -q https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
!wget -q https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h
```

**Step 2 — Clone the repo:**
```bash
!git clone https://github.com/saathviksheerla/gpu_image_pipeline
%cd gpu_image_pipeline
```

**Step 3 — Compile:**
```bash
!nvcc pipeline.cu -o gpu_pipeline
```

**Step 4 — Upload your image and run:**
```bash
!./gpu_pipeline sample_inputs/dog.jpg
```

Download `original.png` and `output.png` from the Colab file browser (left sidebar).

---

## Context

Extended from an assignment in **CS6023: GPU Programming** at IIT Madras - 
added real image I/O and a full demo pipeline on top of the original kernels.

The preprocessing spec (resize 256 -> crop 224, mean 0.485, std 0.229) matches PyTorch's standard `transforms` pipeline:
```python
transforms.Resize(256)
transforms.CenterCrop(224)
transforms.Normalize(mean=[0.485], std=[0.229])
```

This CUDA pipeline replicates that - on raw GPU kernels, no framework.

---

## Why an octopus?

Eight arms. Eight GPU cores. Coincidence? 🐙