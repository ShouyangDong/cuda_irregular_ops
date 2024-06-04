__global__ void __launch_bounds__(1024) sumpool(float* __restrict__ A, float* __restrict__ pool_avg) {
  float pool_sum[1];
  pool_sum[0] = 0.000000e+00f;
  for (int rv0 = 0; rv0 < 5; ++rv0) {
    for (int rv1 = 0; rv1 < 5; ++rv1) {
      pool_sum[0] = (pool_sum[0] + A[(((((((((int)blockIdx.x) / 225) * 262144) + (((((((int)blockIdx.x) % 225) * 4) + (((int)threadIdx.x) >> 8)) / 15) * 4096)) + (rv0 * 4096)) + (rv1 * 64)) + ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 6)) % 60) * 64)) + (((int)threadIdx.x) & 63))]);
    }
  }
  pool_avg[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = pool_sum[0];
}

extern "C" void sumpool_kernel(float *output, float *input, int batch_size, int channels, int input_H, int kernel_size, int stride) {
    float *d_input, *d_output;
    int output_H = (H - kernel_size) / stride + 1;
    int input_size = batch_size * kernel_size * kernel_size * channels;
    int output_size = batch_size * output_H * output_H * channels;
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));

    cudaMemcpy(d_input, input, input_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(128);
    dim3 numBlocks((input_size + blockSize.x - 1) / blockSize.x);

    sumpool<<<numBlocks, blockSize>>>(d_input, d_output);

    cudaMemcpy(output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
