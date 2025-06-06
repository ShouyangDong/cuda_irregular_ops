__global__ void gemv(float *A, float *x, float *y) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < 32) {
    float sum = 0.0f;
    for (int i = 0; i < 512; i++) {
      sum += A[row * 512 + i] * x[i];
    }
    y[row] = sum;
  }
}

extern "C" void gemv_kernel(float *A, float *x, float *y, int m, int n) {
  float *d_A, *d_x, *d_y;

  cudaMalloc(&d_A, m * n * sizeof(float));
  cudaMalloc(&d_x, n * sizeof(float));
  cudaMalloc(&d_y, m * sizeof(float));

  cudaMemcpy(d_A, A, m * n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

  int blockSize = 32;
  int numBlocks = (m + blockSize - 1) / blockSize;

  gemv<<<numBlocks, blockSize>>>(d_A, d_x, d_y);

  cudaMemcpy(y, d_y, m * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_x);
  cudaFree(d_y);
}
