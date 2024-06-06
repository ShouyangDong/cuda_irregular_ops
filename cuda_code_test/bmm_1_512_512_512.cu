__global__ void bmm(float *A, float *B, float *C) {
    int batch_idx = blockIdx.z;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (batch_idx < 1 && row < 512 && col < 512) {
        float sum = 0.0f;
        for (int i = 0; i < 512; i++) {
            sum += A[batch_idx * 512 * 512 + row * 512 + i] * B[batch_idx * 512 * 512 + i * 512 + col];
        }
        C[batch_idx * 512 * 512 + row * 512 + col] = sum;
    }
}

extern "C" void bmm_kernel(float *C, float *A, float *B, int b, int m, int k, int n) {
    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, m * k * sizeof(float));
    cudaMalloc(&d_B, k * n * sizeof(float));
    cudaMalloc(&d_C, m * n * sizeof(float));

    cudaMemcpy(d_A, A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, k * n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(512, 512, 1);
    dim3 numBlocks((m + blockSize.x - 1) / blockSize.x, (n + blockSize.y - 1) / blockSize.y, b + blockSize.z - 1);

    bmm<<<numBlocks, blockSize>>>(d_A, d_B, d_C);

    cudaMemcpy(C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
