/* ==========================================================================
 * GPU Image Preprocessing Pipeline - Demo
 * CS6023: GPU Programming, IIT Madras
 *
 * Loads a real color image, runs 4 CUDA kernels:
 *   grayscale -> resize -> center-crop -> normalize
 * Saves before/after PNGs.
 *
 * Follows standard ImageNet preprocessing:
 *   Resize: 256x256, Crop: 224x224, Mean: 0.485, Std: 0.229
 * ========================================================================== */

/* stb_image: single-header library to load JPG/PNG from disk */
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

/* stb_image_write: single-header library to save PNG to disk */
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

/* Ceiling division macro */
#define CEIL_DIV(n, b) (((n) + (b) - 1) / (b))

/* ITU-R BT.601 luma weights */
#define W_RED   0.299f
#define W_GREEN 0.587f
#define W_BLUE  0.114f

/* Standard ImageNet preprocessing values */
#define RESIZE_H 256
#define RESIZE_W 256
#define CROP_H   224
#define CROP_W   224
#define MEAN     0.485f
#define STD      0.229f

/* CUDA error checking */
#define CUDA_CHECK(call) \
    if ((call) != cudaSuccess) { \
        fprintf(stderr, "CUDA error at line %d: %s\n", __LINE__, \
                cudaGetErrorString(cudaGetLastError())); \
        exit(1); \
    }

/* ------------------------------------------------------------------
 * KERNEL 1: Grayscale
 * One thread per pixel.
 * RGB interleaved -> single float using weighted luma formula.
 * ------------------------------------------------------------------ */
__global__ void grayscaleKernel(const unsigned char *d_rgb,
                                float *d_gray, int H, int W)
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= H * W) return;

    /* Pixel idx: R is at 3*idx, G at 3*idx+1, B at 3*idx+2 */
    float R = (float)d_rgb[3 * idx];
    float G = (float)d_rgb[3 * idx + 1];
    float B = (float)d_rgb[3 * idx + 2];

    d_gray[idx] = W_RED * R + W_GREEN * G + W_BLUE * B;
}

/* ------------------------------------------------------------------
 * KERNEL 2: Bilinear Resize
 * One thread per output pixel.
 * Maps each output pixel back to a fractional input coordinate,
 * then blends the 4 surrounding input pixels.
 * ------------------------------------------------------------------ */
__global__ void resizeKernel(const float *d_gray, float *d_resized,
                             int H, int W, int Hr, int Wr)
{
    unsigned ox = blockIdx.x * blockDim.x + threadIdx.x; /* output col */
    unsigned oy = blockIdx.y * blockDim.y + threadIdx.y; /* output row */
    if (oy >= Hr || ox >= Wr) return;

    /* Scale: map output coords -> input coords (align corners) */
    float scaleY = (Hr > 1) ? (float)(H - 1) / (float)(Hr - 1) : 0.0f;
    float scaleX = (Wr > 1) ? (float)(W - 1) / (float)(Wr - 1) : 0.0f;

    float fy = oy * scaleY; /* exact input row, as float */
    float fx = ox * scaleX; /* exact input col, as float */

    /* 4 nearest input pixels */
    int y0 = (int)floorf(fy);
    int x0 = (int)floorf(fx);
    int y1 = min(y0 + 1, H - 1);
    int x1 = min(x0 + 1, W - 1);

    /* Fractional weights */
    float wy = fy - y0;
    float wx = fx - x0;

    /* Bilinear blend */
    float top    = d_gray[y0*W + x0]*(1-wx) + d_gray[y0*W + x1]*wx;
    float bottom = d_gray[y1*W + x0]*(1-wx) + d_gray[y1*W + x1]*wx;
    d_resized[oy * Wr + ox] = top*(1-wy) + bottom*wy;
}

/* ------------------------------------------------------------------
 * KERNEL 3: Center Crop
 * One thread per output pixel.
 * Skips equal margins on each side - pure index offset, no math.
 * ------------------------------------------------------------------ */
__global__ void cropKernel(const float *d_resized, float *d_cropped,
                           int Hr, int Wr, int Hc, int Wc)
{
    unsigned ox = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= Wc || oy >= Hc) return;

    /* How many pixels to skip from each edge */
    int offsetX = (Wr - Wc) / 2;
    int offsetY = (Hr - Hc) / 2;

    d_cropped[oy*Wc + ox] = d_resized[(oy + offsetY)*Wr + (ox + offsetX)];
}

/* ------------------------------------------------------------------
 * KERNEL 4: Normalize
 * One thread per pixel.
 * Converts 0-255 -> zero-centered floats expected by neural networks.
 * Formula: (pixel/255 - mean) / std
 * ------------------------------------------------------------------ */
__global__ void normalizeKernel(const float *d_cropped, float *d_out,
                                int Hc, int Wc, float mean, float stdv)
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Hc * Wc) return;

    d_out[idx] = (d_cropped[idx] / 255.0f - mean) / stdv;
}

/* ================================================================== */
int main(int argc, char **argv)
{
    /* ----------------------------------------------------------
     * LOAD IMAGE
     * stbi_load returns flat RGB array: R,G,B,R,G,B,...
     * Last arg = 3 forces 3 channels even if image is RGBA.
     * ---------------------------------------------------------- */
    if (argc < 2) {
    fprintf(stderr, "Usage: ./gpu_pipeline <image_path>\n");
    return 1;
    }
    int W, H, channels;
    unsigned char *h_rgb = stbi_load(argv[1], &W, &H, &channels, 3);
    if (!h_rgb) {
        fprintf(stderr, "ERROR: could not load %s\n", argv[1]);
        return 1;
    }
    printf("Loaded: dog.jpg (%d x %d, %d channels)\n", W, H, channels);

    /* Save original color image for README before/after comparison */
    stbi_write_png("original.png", W, H, 3, h_rgb, W * 3);
    printf("Saved: original.png\n");

    /* Sizes for each pipeline stage */
    int Hr = RESIZE_H, Wr = RESIZE_W;
    int Hc = CROP_H,   Wc = CROP_W;
    size_t nPixIn   = (size_t)H  * W;
    size_t nRgb     = nPixIn * 3;
    size_t nPixRes  = (size_t)Hr * Wr;
    size_t nPixCrop = (size_t)Hc * Wc;

    /* Host buffer to receive cropped result from GPU */
    float *h_cropped = (float *)malloc(nPixCrop * sizeof(float));

    /* ----------------------------------------------------------
     * ALLOCATE GPU MEMORY - one buffer per pipeline stage
     * ---------------------------------------------------------- */
    unsigned char *d_rgb;
    float *d_gray, *d_resized, *d_cropped, *d_out;

    CUDA_CHECK(cudaMalloc(&d_rgb,     nRgb     * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_gray,    nPixIn   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_resized, nPixRes  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cropped, nPixCrop * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out,     nPixCrop * sizeof(float)));

    /* ----------------------------------------------------------
     * COPY INPUT: CPU -> GPU
     * ---------------------------------------------------------- */
    CUDA_CHECK(cudaMemcpy(d_rgb, h_rgb, nRgb * sizeof(unsigned char),
                          cudaMemcpyHostToDevice));

    /* ----------------------------------------------------------
     * LAUNCH KERNELS in pipeline order
     * ---------------------------------------------------------- */
    unsigned blockSize = 256;

    /* Stage 1: Grayscale - 1D grid */
    grayscaleKernel<<<CEIL_DIV(H*W, blockSize), blockSize>>>(
        d_rgb, d_gray, H, W);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("Done: grayscale\n");

    /* Stage 2: Resize - 2D grid (one thread per output pixel) */
    dim3 block2D(16, 16);
    resizeKernel<<<dim3(CEIL_DIV(Wr,16), CEIL_DIV(Hr,16)), block2D>>>(
        d_gray, d_resized, H, W, Hr, Wr);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("Done: resize %dx%d -> %dx%d\n", H, W, Hr, Wr);

    /* Stage 3: Center Crop - 2D grid */
    cropKernel<<<dim3(CEIL_DIV(Wc,16), CEIL_DIV(Hc,16)), block2D>>>(
        d_resized, d_cropped, Hr, Wr, Hc, Wc);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("Done: center crop -> %dx%d\n", Hc, Wc);

    /* Stage 4: Normalize - 1D grid */
    normalizeKernel<<<CEIL_DIV(Hc*Wc, blockSize), blockSize>>>(
        d_cropped, d_out, Hc, Wc, MEAN, STD);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("Done: normalize (mean=%.3f, std=%.3f)\n", MEAN, STD);

    /* ----------------------------------------------------------
     * COPY RESULT: GPU -> CPU
     * We use d_cropped (0-255 range) for the output PNG.
     * d_out (normalized) has values like -1.2, 0.4 - not visual.
     * ---------------------------------------------------------- */
    CUDA_CHECK(cudaMemcpy(h_cropped, d_cropped, nPixCrop * sizeof(float),
                          cudaMemcpyDeviceToHost));

    /* Convert float -> unsigned char for PNG */
    unsigned char *h_out_uchar = (unsigned char *)malloc(nPixCrop);
    for (size_t i = 0; i < nPixCrop; i++) {
        float v = h_cropped[i];
        if (v < 0.0f)   v = 0.0f;   /* clamp low */
        if (v > 255.0f) v = 255.0f; /* clamp high */
        h_out_uchar[i] = (unsigned char)v;
    }

    /* 1 channel = grayscale PNG */
    stbi_write_png("output.png", Wc, Hc, 1, h_out_uchar, Wc);
    printf("Saved: output.png (%dx%d grayscale)\n", Wc, Hc);

    /* ----------------------------------------------------------
     * CLEANUP
     * ---------------------------------------------------------- */
    cudaFree(d_rgb);
    cudaFree(d_gray);
    cudaFree(d_resized);
    cudaFree(d_cropped);
    cudaFree(d_out);
    stbi_image_free(h_rgb);
    free(h_cropped);
    free(h_out_uchar);

    printf("\nPipeline complete. Download original.png and output.png.\n");
    return 0;
}
