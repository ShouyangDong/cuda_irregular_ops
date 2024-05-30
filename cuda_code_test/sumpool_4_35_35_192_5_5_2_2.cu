__global__ void __launch_bounds__(1024) sumpool_kernel(float* __restrict__ A, float* __restrict__ pool_avg) {
  float pool_sum[1];
  pool_sum[0] = 0.000000e+00f;
  for (int rv0 = 0; rv0 < 5; ++rv0) {
    for (int rv1 = 0; rv1 < 5; ++rv1) {
      pool_sum[0] = (pool_sum[0] + A[(((((((((int)blockIdx.x) / 48) * 235200) + (((((int)blockIdx.x) % 48) / 3) * 13440)) + (rv0 * 6720)) + (((((((int)blockIdx.x) % 3) * 16) + (((int)threadIdx.x) >> 6)) / 3) * 384)) + (rv1 * 192)) + (((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) % 192))]);
    }
  }
  pool_avg[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = pool_sum[0];
}