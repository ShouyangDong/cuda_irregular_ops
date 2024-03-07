#include <cuda_runtime.h>
#include "stdio.h"


__global__ void cuda_rms_norm(float* A, float* B, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float eps = 1e-5f;
    
    if (idx < size) {
        // Calculate sum
        float sum = 0.0;
        for (int j = 0; j < size; j++) {
            sum += A[idx * size + j] * A[idx * size + j];
        }

        // Calculate mean
        float mean = sum / size;

        // Calculate scale
        float scale = 1.0 / sqrt(mean + eps);

        // Normalize and store in B
        for (int j = 0; j < size; j++) {
            B[idx * size + j] = A[idx * size + j] * scale;
        }
    }
}


extern "C" void rms_norm_kernel(float* A, float* B) {
    // Allocate memory on the device
    float *d_A, *d_B;
    int size = 8192;
    int num_elements = size * size;
    cudaMalloc(&d_A, num_elements * sizeof(float));
    cudaMalloc(&d_B, num_elements * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, A, num_elements * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    // Launch kernel
    cuda_rms_norm<<<num_blocks, block_size>>>(d_A, d_B, size);

    // Copy the result back to host
    cudaMemcpy(B, d_B, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
}
