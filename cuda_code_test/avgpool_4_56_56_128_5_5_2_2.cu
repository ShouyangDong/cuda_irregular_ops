__global__ void __launch_bounds__(1024) avgpool_kernel(float* __restrict__ A, float* __restrict__ pool_avg) {
  float pool_sum[1];
  pool_sum[0] = 0.000000e+00f;
  for (int rv0 = 0; rv0 < 5; ++rv0) {
    for (int rv1 = 0; rv1 < 5; ++rv1) {
      pool_sum[0] = (pool_sum[0] + A[(((((((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 9)) / 169) * 401408) + (((((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 8)) % 338) / 13) * 14336)) + (rv0 * 7168)) + ((((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 7)) % 26) * 256)) + (rv1 * 128)) + (((int)threadIdx.x) & 127))]);
    }
  }
  pool_avg[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (pool_sum[0] * 4.000000e-02f);
}