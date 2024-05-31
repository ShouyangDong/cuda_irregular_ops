__global__ void __launch_bounds__(12) softmax(float* __restrict__ A, float* __restrict__ T_softmax_maxelem) {
  if (threadIdx.x < 12) {
      
      float maxVal = A[threadIdx.x * 5];
      for (int i = 1; i < 5; ++i) {
          if (A[threadIdx.x * 5 + i] > maxVal) {
              maxVal = A[threadIdx.x * 5 + i];
          }
      }
      
      
      float denom = 0.0f;
      for (int i = 0; i < 5; ++i) {
          T_softmax_maxelem[threadIdx.x * 5 + i] = expf(A[threadIdx.x * 5 + i] - maxVal);
          denom += T_softmax_maxelem[threadIdx.x * 5 + i];
      }
      
      
      for (int i = 0; i < 5; ++i) {
          T_softmax_maxelem[threadIdx.x * 5 + i] /= denom;
      }
  }
}

extern "C" void softmax_kernel(float *C, float *A, int size) {
  float *d_A, *d_C;

  cudaMalloc(&d_A, size * sizeof(float));
  cudaMalloc(&d_C, size * sizeof(float));

  cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockSize(1024);
  dim3 numBlocks((size + 1024 - 1) / 1024);

  softmax<<<numBlocks, blockSize>>>(d_A, d_C);

  cudaMemcpy(C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_C);
}
