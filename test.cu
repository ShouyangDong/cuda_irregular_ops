#include <cuda_runtime.h>
#include <random>
#include <vector>
#include <cmath>
#include <iostream>

const int BLOCK_SIZE = 16;

__global__ void fusedScaledDotProductAttention(
    const float* Q, 
    const float* K,
    const float* V,
    float* attention, 
    int sequenceLength, 
    int dim
    ) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ float expValues[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sumOfExponents[BLOCK_SIZE];

    if (threadIdx.x == 0) {
        sumOfExponents[threadIdx.y] = 0.0f;
    }

    __syncthreads();

    if (row < sequenceLength && col < dim) {
        float dotProduct = 0.0f;

        // Compute the dot product Q * K^T
        for (int idx = 0; idx < dim; ++idx) {
            dotProduct += Q[row * dim + idx] * K[idx * sequenceLength + col];
        }

        // Scale the dot product
        dotProduct /= sqrtf(static_cast<float>(dim));

        // Apply exponential function
        float expValue = expf(dotProduct);
        expValues[threadIdx.y][threadIdx.x] = expValue;

        // Accumulate the sum of exponentials
        atomicAdd(&sumOfExponents[threadIdx.y], expValue);
    }

    __syncthreads();

    if (row < sequenceLength && col < dim) {
        // Normalize the exponentiated values
        float softmaxValue = expValues[threadIdx.y][threadIdx.x] / sumOfExponents[threadIdx.y];

        // Compute the final attention value
        float finalValue = 0.0f;
        for (int idx = 0; idx < dim; ++idx) {
            finalValue += softmaxValue * V[idx * sequenceLength + col];
        }

        // Store the final attention value
        attention[row * dim + col] = finalValue;
    }
}

void fillMatrix(float* matrix, int rows, int cols) {
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            matrix[row * cols + col] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
}

int main() {
    int sequenceLength = 512;
    int dim = 1024;
    size_t size = sequenceLength * dim * sizeof(float);

    float *queryProjection, *keyProjection, *valueProjection, *attention;
    cudaMalloc(&queryProjection, size);
    cudaMalloc(&keyProjection, size);
    cudaMalloc(&valueProjection, size);
    cudaMalloc(&attention, size);

    // Create host arrays and fill them
    float *hostQueryProjection = (float*)malloc(size);
    float *hostKeyProjection = (float*)malloc(size);
    float *hostValueProjection = (float*)malloc(size);
    
    fillMatrix(hostQueryProjection, sequenceLength, dim);
    fillMatrix(hostKeyProjection, sequenceLength, dim);
    fillMatrix(hostValueProjection, sequenceLength, dim);

    // Copy data from host to device
    cudaMemcpy(queryProjection, hostQueryProjection, size, cudaMemcpyHostToDevice);
    cudaMemcpy(keyProjection, hostKeyProjection, size, cudaMemcpyHostToDevice);
    cudaMemcpy(valueProjection, hostValueProjection, size, cudaMemcpyHostToDevice);

    // Calculate grid and block dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((dim + BLOCK_SIZE - 1) / BLOCK_SIZE, (sequenceLength + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch the kernel
    fusedScaledDotProductAttention<<<gridDim, blockDim>>>(
        queryProjection,
        keyProjection,
        valueProjection,
        attention,
        sequenceLength,
        dim
    );

    // Check for errors
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
    }

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Free device memory
    cudaFree(queryProjection);
    cudaFree(keyProjection);
    cudaFree(valueProjection);
    cudaFree(attention);

    // Free host memory
    free(hostQueryProjection);
    free(hostKeyProjection);
    free(hostValueProjection);

    return 0;
}