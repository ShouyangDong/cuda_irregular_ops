__global__ void __launch_bounds__(5) softmax(float* __restrict__ A, float* __restrict__ T_softmax_norm) {
    if (threadIdx.x < 5) {
    int rowStart = threadIdx.x * 128;
    
    float maxVal = A[rowStart];
    for (int i = 1; i < 128; ++i) {
        if (A[rowStart + i] > maxVal) {
            maxVal = A[rowStart + i];
        }
    }
    
    float denom = 0.0f;
    for (int i = 0; i < 128; ++i) {
        T_softmax_norm[rowStart + i] = expf(A[rowStart + i] - maxVal);
        denom += T_softmax_norm[rowStart + i];
    }
    
    for (int i = 0; i < 128; ++i) {
        T_softmax_norm[rowStart + i] /= denom;
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
