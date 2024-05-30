extern "C" __global__ void __launch_bounds__(60) relu(float* __restrict__ A, float* __restrict__ compute) {
  compute[((int)threadIdx.x)] = max(A[((int)threadIdx.x)], 0.000000e+00f);
}

extern "C" void relu_kernel(float *C, float *A, int size) {
  float *d_A, *d_C;

  cudaMalloc(&d_A, size * sizeof(float));
  cudaMalloc(&d_C, size * sizeof(float));

  cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockSize(1024);
  dim3 numBlocks((size + 1024 - 1) / 1024);

  relu<<<numBlocks, blockSize>>>(d_A, d_C);

  cudaMemcpy(C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_C);
}
