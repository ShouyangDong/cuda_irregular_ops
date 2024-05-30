__global__ void __launch_bounds__(1024) sumpool_kernel(float* __restrict__ A, float* __restrict__ pool_avg) {
  float pool_sum[1];
  pool_sum[0] = 0.000000e+00f;
  for (int rv0 = 0; rv0 < 3; ++rv0) {
    for (int rv1 = 0; rv1 < 3; ++rv1) {
      if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 2309) {
        pool_sum[0] = (pool_sum[0] + A[(((((((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 6)) / 9) * 4096) + (((((((int)blockIdx.x) * 7) + (((int)threadIdx.x) >> 6)) % 9) / 3) * 1024)) + (rv0 * 512)) + ((((((int)threadIdx.x) >> 6) + ((int)blockIdx.x)) % 3) * 128)) + (rv1 * 64)) + (((int)threadIdx.x) & 63))]);
      }
    }
  }
  if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 2309) {
    pool_avg[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = pool_sum[0];
  }
}