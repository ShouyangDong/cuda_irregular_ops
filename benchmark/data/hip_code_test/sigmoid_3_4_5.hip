__global__ void __launch_bounds__(60)
    sigmoid(float *__restrict__ A, float *__restrict__ compute) {
  compute[((int)threadIdx.x)] =
      (1.000000e+00f /
       (1.000000e+00f + __expf((0.000000e+00f - A[((int)threadIdx.x)]))));
}

extern "C" void sigmoid_kernel(float *A, float *C, int size) {
  float *d_A;
  float *d_C;

  hipMalloc(&d_A, size * sizeof(float));
  hipMalloc(&d_C, size * sizeof(float));

  hipMemcpy(d_A, A, size * sizeof(float), hipMemcpyHostToDevice);

  dim3 blockSize(60);
  dim3 numBlocks((size + 60 - 1) / 60);

  sigmoid<<<numBlocks, blockSize>>>(d_A, d_C);

  hipMemcpy(C, d_C, size * sizeof(float), hipMemcpyDeviceToHost);

  hipFree(d_A);
  hipFree(d_C);
}
