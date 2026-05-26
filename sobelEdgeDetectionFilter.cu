#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

using namespace cv;

// ─────────────────────────────────────────────────────────────────────────────
// CUDA Kernel: sobelFilter
//   • Converts each pixel to grayscale (if colour) using luminosity weights
//   • Applies Sobel Gx and Gy kernels
//   • Boundary-checks so edge pixels are handled without out-of-bounds reads
// ─────────────────────────────────────────────────────────────────────────────
__global__ void sobelFilter(unsigned char *srcImage,
                             unsigned char *dstImage,
                             unsigned int   width,
                             unsigned int   height)
{
    // Compute the (x, y) position this thread handles
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check – skip threads that fall outside the image
    if (x >= width || y >= height) return;

    // Sobel kernels
    //   Gx:  [-1  0  1]      Gy:  [-1 -2 -1]
    //        [-2  0  2]           [ 0  0  0]
    //        [-1  0  1]           [ 1  2  1]
    int Gx = 0, Gy = 0;

    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            // Clamp neighbour coordinates to valid range (handles image borders)
            int nx = min(max(x + kx, 0), (int)width  - 1);
            int ny = min(max(y + ky, 0), (int)height - 1);

            // The source image is stored as BGR (3 channels); convert to grey
            int idx   = (ny * width + nx) * 3;   // 3 bytes per pixel (BGR)
            int blue  = srcImage[idx    ];
            int green = srcImage[idx + 1];
            int red   = srcImage[idx + 2];
            // Luminosity / BT.601 weights
            int grey  = (int)(0.114f * blue + 0.587f * green + 0.299f * red);

            // Sobel weights for this kernel position
            int wx = (kx == -1 ? -1 : kx == 1 ? 1 : 0) * (ky == 0 ? 2 : 1);
            int wy = (ky == -1 ? -1 : ky == 1 ? 1 : 0) * (kx == 0 ? 2 : 1);

            Gx += wx * grey;
            Gy += wy * grey;
        }
    }

    // Gradient magnitude, clamped to [0, 255]
    int magnitude = (int)sqrtf((float)(Gx * Gx + Gy * Gy));
    dstImage[y * width + x] = (unsigned char)min(magnitude, 255);
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: check CUDA return codes
// ─────────────────────────────────────────────────────────────────────────────
void checkCudaErrors(cudaError_t r) {
    if (r != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(r));
        exit(EXIT_FAILURE);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    // ── 1. Read input image (colour) ──────────────────────────────────────────
    Mat image = imread("/content/images.jpeg", IMREAD_COLOR);
    if (image.empty()) {
        printf("Error: Image not found.\n");
        return -1;
    }

    int width     = image.cols;
    int height    = image.rows;
    // Input: 3-channel BGR  |  Output: 1-channel grayscale edge map
    size_t inputSize  = (size_t)width * height * 3 * sizeof(unsigned char);
    size_t outputSize = (size_t)width * height     * sizeof(unsigned char);

    printf("Image size: %d x %d\n", width, height);

    // ── 2. Allocate host output buffer ────────────────────────────────────────
    unsigned char *h_outputImage = (unsigned char *)malloc(outputSize);
    if (!h_outputImage) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return -1;
    }

    // ── 3. Allocate device memory and copy input ──────────────────────────────
    unsigned char *d_inputImage, *d_outputImage;
    checkCudaErrors(cudaMalloc(&d_inputImage,  inputSize));
    checkCudaErrors(cudaMalloc(&d_outputImage, outputSize));
    checkCudaErrors(cudaMemcpy(d_inputImage, image.data, inputSize,
                               cudaMemcpyHostToDevice));

    // ── 4. CUDA timing events ─────────────────────────────────────────────────
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ── 5. Launch kernel (16×16 thread block) ─────────────────────────────────
    dim3 blockSize(16, 16);
    dim3 gridSize((int)ceil(width  / 16.0),
                  (int)ceil(height / 16.0));

    cudaEventRecord(start);
    sobelFilter<<<gridSize, blockSize>>>(d_inputImage, d_outputImage,
                                         width, height);
    cudaEventRecord(stop);

    // ── 6. Check for kernel launch errors ────────────────────────────────────
    checkCudaErrors(cudaGetLastError());
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("CUDA Sobel kernel execution time: %.4f ms\n", milliseconds);

    // ── 7. Copy result back to host ───────────────────────────────────────────
    checkCudaErrors(cudaMemcpy(h_outputImage, d_outputImage, outputSize,
                               cudaMemcpyDeviceToHost));

    // ── 8. Write output image ─────────────────────────────────────────────────
    Mat outputImage(height, width, CV_8UC1, h_outputImage);
    imwrite("output_sobel.jpeg", outputImage);
    printf("Output written to output_sobel.jpeg\n");

    // ── 9. Free resources ─────────────────────────────────────────────────────
    free(h_outputImage);
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
