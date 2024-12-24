__global__ void __launch_bounds__(105)
    softmax(float *__restrict__ A, float *__restrict__ T_softmax_maxelem) {
  if (threadIdx.x < 105) {

    float maxVal = A[threadIdx.x * 32];
    for (int i = 1; i < 32; ++i) {
      if (A[threadIdx.x * 32 + i] > maxVal) {
        maxVal = A[threadIdx.x * 32 + i];
      }
    }

    float denom = 0.0f;
    for (int i = 0; i < 32; ++i) {
      T_softmax_maxelem[threadIdx.x * 32 + i] =
          expf(A[threadIdx.x * 32 + i] - maxVal);
      denom += T_softmax_maxelem[threadIdx.x * 32 + i];
    }

    for (int i = 0; i < 32; ++i) {
      T_softmax_maxelem[threadIdx.x * 32 + i] /= denom;
    }
  }
}

extern "C" void softmax_kernel(float *C, float *A, int size1, int size2) {
  float *d_A;
  float *d_C;

  cudaMalloc(&d_A, size1 * size2 * sizeof(float));
  cudaMalloc(&d_C, size1 * size2 * sizeof(float));

  cudaMemcpy(d_A, A, size1 * size2 * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockSize(105);
  dim3 numBlocks((size1 + 105 - 1) / 105);

  for (int i = 0; i < 10; i++) {
    softmax<<<numBlocks, blockSize>>>(d_A, d_C);
  }

  // 定义 CUDA 事件以计算时间
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // 启动内核
  cudaEventRecord(start);
  for (int i = 0; i < 1000; ++i) {
    softmax<<<numBlocks, blockSize>>>(d_A, d_C);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // 计算执行时间
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  milliseconds = milliseconds / 1000.0f;
  printf("Execution time: %f milliseconds\n", milliseconds);

  cudaMemcpy(C, d_C, size1 * size2 * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_C);
}
