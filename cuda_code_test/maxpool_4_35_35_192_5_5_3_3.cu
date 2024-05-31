__global__ void __launch_bounds__(1024) maxpool(float* __restrict__ A, float* __restrict__ pool_max) {
  float pool_max_local[1];
  pool_max_local[0] = -3.402823e+38f;
  for (int rv0 = 0; rv0 < 5; ++rv0) {
    for (int rv1 = 0; rv1 < 5; ++rv1) {
      if (((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 8)) < 363) {
        pool_max_local[0] = max(pool_max_local[0], A[(((((((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 6)) / 363) * 235200) + (((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 6)) % 363) / 33) * 20160)) + (rv0 * 6720)) + (((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 6)) % 33) / 3) * 576)) + (rv1 * 192)) + (((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) % 192))]);
      }
    }
  }
  if (((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 8)) < 363) {
    pool_max[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = pool_max_local[0];
  }
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
