// Forward declaration of the device function
__device__ float geluf(float x);

__device__ float geluf(float x) {
  return 0.5 * x * (1 + tanh(sqrt(2 / M_PI) * (x + 0.044715 * pow(x, 3))));
}

__global__ void __launch_bounds__(60)
    gelu(float *__restrict__ A, float *__restrict__ compute) {
  compute[((int)threadIdx.x)] = geluf(A[((int)threadIdx.x)]);
}

extern "C" void gelu_kernel(float *C, float *A, int size) {
  float *d_A;
  float *d_C;

  cudaMalloc(&d_A, size * sizeof(float));
  cudaMalloc(&d_C, size * sizeof(float));

  cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockSize(60);
  dim3 numBlocks((size + 60 - 1) / 60);

  for (int i =0; i< 10; i++){
    gelu<<<numBlocks, blockSize>>>(d_A, d_C);
  }
  
  // 定义 CUDA 事件以计算时间
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // 启动内核
  cudaEventRecord(start);
  for (int i = 0; i < 1000; ++i) {
      gelu<<<numBlocks, blockSize>>>(d_A, d_C);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // 计算执行时间
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  milliseconds = milliseconds / 1000.0f;
  printf("Execution time: %f milliseconds\n", milliseconds);
  cudaMemcpy(C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_C);
}
