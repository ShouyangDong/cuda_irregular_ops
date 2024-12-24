__global__ void gemv(float *y, float *A, float *x) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < 3) {
    float sum = 0.0f;
    for (int i = 0; i < 512; i++) {
      sum += A[row * 512 + i] * x[i];
    }
    y[row] = sum;
  }
}

extern "C" void gemv_kernel(float *y, float *A, float *x, int m, int n) {
  float *d_A, *d_x, *d_y;

  cudaMalloc(&d_A, m * n * sizeof(float));
  cudaMalloc(&d_x, n * sizeof(float));
  cudaMalloc(&d_y, m * sizeof(float));

  cudaMemcpy(d_A, A, m * n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

  int blockSize = 3;
  int numBlocks = (m + blockSize - 1) / blockSize;

  for (int i = 0; i < 1000; ++i) {
    gemv<<<numBlocks, blockSize>>>(d_y, d_A, d_x);
  }
  // 定义 CUDA 事件以计算时间
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // 启动内核
  cudaEventRecord(start);
  for (int i = 0; i < 1000; ++i) {
    gemv<<<numBlocks, blockSize>>>(d_y, d_A, d_x);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // 计算执行时间
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  milliseconds = milliseconds / 1000.0f;
  printf("Execution time: %f milliseconds\n", milliseconds);

  cudaMemcpy(y, d_y, m * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_x);
  cudaFree(d_y);
}
