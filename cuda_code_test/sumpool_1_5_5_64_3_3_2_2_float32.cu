extern "C" __global__ void __launch_bounds__(1024) sumpool_kernel(float* __restrict__ A, float* __restrict__ pool_avg) {
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