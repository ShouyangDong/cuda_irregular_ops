#include <cuda_runtime.h>
#include "stdio.h"


__global__ void cuda_layer_norm(
    float* A, 
    float* gamma, 
    float* beta, 
    float* B, 
    int batch_size,
    int seq_lenght,
    int d_model) 
{
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


extern "C" void layer_norm_kernel(float* A, float* gamma, float* beta, float* B) {
    // Allocate memory on the device
    float *d_A, *d_B, *d_gamma, *d_beta;
    int batch_size = 2;
    int seq_lenght = 4;
    int d_model = 8;
    int num_elements = batch_size * seq_lenght * d_model;
    cudaMalloc(&d_A, num_elements * sizeof(float));
    cudaMalloc(&d_B, num_elements * sizeof(float));
    cudaMalloc(&d_gamma, d_model * sizeof(float));
    cudaMalloc(&d_beta, d_model * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, A, num_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, gamma, d_model * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, beta, d_model * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;

    // Launch kernel
    cuda_layer_norm<<<num_blocks, block_size>>>(d_A, d_gamma, d_beta, d_B, size);

    // Copy the result back to host
    cudaMemcpy(B, d_B, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_gamma);
    cudaFree(d_beta);
}
