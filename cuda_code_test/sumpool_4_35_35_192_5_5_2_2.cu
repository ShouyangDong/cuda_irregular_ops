__global__ void __launch_bounds__(1024) sumpool(float* __restrict__ A, float* __restrict__ pool_avg) {
  float pool_sum[1];
  pool_sum[0] = 0.000000e+00f;
  for (int rv0 = 0; rv0 < 5; ++rv0) {
    for (int rv1 = 0; rv1 < 5; ++rv1) {
      pool_sum[0] = (pool_sum[0] + A[(((((((((int)blockIdx.x) / 48) * 235200) + (((((int)blockIdx.x) % 48) / 3) * 13440)) + (rv0 * 6720)) + (((((((int)blockIdx.x) % 3) * 16) + (((int)threadIdx.x) >> 6)) / 3) * 384)) + (rv1 * 192)) + (((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) % 192))]);
    }
  }
  pool_avg[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = pool_sum[0];
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
