__global__ void __launch_bounds__(1024) maxpool(float* __restrict__ A, float* __restrict__ pool_max) {
  float pool_max_local[1];
  pool_max_local[0] = -3.402823e+38f;
  for (int rv0 = 0; rv0 < 5; ++rv0) {
    for (int rv1 = 0; rv1 < 5; ++rv1) {
      pool_max_local[0] = max(pool_max_local[0], A[(((((((((int)threadIdx.x) >> 8) * 4096) + (((((int)threadIdx.x) & 255) >> 7) * 1536)) + (rv0 * 512)) + (((((int)threadIdx.x) & 127) >> 6) * 192)) + (rv1 * 64)) + (((int)threadIdx.x) & 63))]);
    }
  }
  pool_max[((int)threadIdx.x)] = pool_max_local[0];
}

extern "C" void maxpool_kernel(float *output, float *input, int input_size, int kernel_size, int stride) {
    int input_size = 128;
    int kernel_size = 3;
    float *d_input, *d_output;

    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_output, input_size * sizeof(float));

    cudaMemcpy(d_input, input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(128);
    dim3 numBlocks((input_size + blockSize.x - 1) / blockSize.x);

    maxpool<<<numBlocks, blockSize>>>(d_input, d_kernel, d_output);

    cudaMemcpy(output, d_output, input_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
