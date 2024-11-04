__global__ void bmm(float *A, float *B, float *C) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  for (int batch_idx = 0; batch_idx < 4; ++batch_idx) {
    if (row < 256 && col < 256) {
      float sum = 0.0f;
      for (int i = 0; i < 256; i++) {
        sum += A[batch_idx * 256 * 256 + row * 256 + i] *
               B[batch_idx * 256 * 256 + i * 256 + col];
      }
      C[batch_idx * 256 * 256 + row * 256 + col] = sum;
    }
  }
}

extern "C" void bmm_kernel(float *C, float *A, float *B, int b, int m, int k,
                           int n) {
  float* d_A;
  float* d_B;
  float* d_C;

  cudaMalloc(&d_A, b * m * k * sizeof(float));
  cudaMalloc(&d_B, b * k * n * sizeof(float));
  cudaMalloc(&d_C, b * m * n * sizeof(float));

  cudaMemcpy(d_A, A, b * m * k * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, b * k * n * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockSize(32, 32);
  dim3 numBlocks((m + blockSize.x - 1) / blockSize.x,
                 (n + blockSize.y - 1) / blockSize.y);

  bmm<<<numBlocks, blockSize>>>(d_A, d_B, d_C);

  cudaMemcpy(C, d_C, b * m * n * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}
