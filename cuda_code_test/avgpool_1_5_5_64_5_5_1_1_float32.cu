extern "C" __global__ void __launch_bounds__(1024) avgpool_kernel(float* __restrict__ A, float* __restrict__ pool_avg) {
  float pool_sum[1];
  pool_sum[0] = 0.000000e+00f;
  for (int rv0 = 0; rv0 < 5; ++rv0) {
    for (int rv1 = 0; rv1 < 5; ++rv1) {
      if (((int)threadIdx.x) < 64) {
        pool_sum[0] = (pool_sum[0] + A[(((rv0 * 320) + (rv1 * 64)) + ((int)threadIdx.x))]);
      }
    }
  }
  if (((int)threadIdx.x) < 64) {
    pool_avg[((int)threadIdx.x)] = (pool_sum[0] * 4.000000e-02f);
  }
}