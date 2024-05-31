__global__ void __launch_bounds__(1024) sumpool(float* __restrict__ A, float* __restrict__ pool_avg) {
  float pool_sum[1];
  pool_sum[0] = 0.000000e+00f;
  for (int rv0 = 0; rv0 < 3; ++rv0) {
    for (int rv1 = 0; rv1 < 3; ++rv1) {
      if (((int)threadIdx.x) < 256) {
        pool_sum[0] = (pool_sum[0] + A[((((((((int)threadIdx.x) >> 7) * 640) + (rv0 * 320)) + (((((int)threadIdx.x) & 127) >> 6) * 128)) + (rv1 * 64)) + (((int)threadIdx.x) & 63))]);
      }
    }
  }
  if (((int)threadIdx.x) < 256) {
    pool_avg[((int)threadIdx.x)] = pool_sum[0];
  }
}

extern "C" void sumpool_kernel(float *output, float *input, int input_size, int kernel_size, int stride) {
    int input_size = 128;
    int kernel_size = 3;
    float *d_input, *d_output;

    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_output, input_size * sizeof(float));

    cudaMemcpy(d_input, input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(128);
    dim3 numBlocks((input_size + blockSize.x - 1) / blockSize.x);

    sumpool<<<numBlocks, blockSize>>>(d_input, d_kernel, d_output);

    cudaMemcpy(output, d_output, input_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}