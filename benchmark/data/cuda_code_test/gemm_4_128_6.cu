__global__ void gemm(float *A, float *B, float *C) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < 4 && col < 6) {
    float sum = 0.0f;
    for (int i = 0; i < 128; i++) {
      sum += A[row * 128 + i] * B[i * 6 + col];
    }
    C[row * 6 + col] = sum;
  }
}

extern "C" void gemm_kernel(float *C, float *A, float *B, int m, int k, int n) {
  float *d_A;
  float *d_B;
  float *d_C;

  cudaMalloc(&d_A, m * k * sizeof(float));
  cudaMalloc(&d_B, k * n * sizeof(float));
  cudaMalloc(&d_C, m * n * sizeof(float));

  cudaMemcpy(d_A, A, m * k * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, k * n * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockSize(4, 6);
  dim3 numBlocks((m + blockSize.x - 1) / blockSize.x,
                 (n + blockSize.y - 1) / blockSize.y);

  gemm<<<numBlocks, blockSize>>>(d_A, d_B, d_C);

  cudaMemcpy(C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}
