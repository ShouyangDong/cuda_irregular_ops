__global__ void __launch_bounds__(1024) minpool_kernel(float* __restrict__ A, float* __restrict__ pool_min) {
  float pool_min_local[1];
  pool_min_local[0] = 3.402823e+38f;
  for (int rv0 = 0; rv0 < 5; ++rv0) {
    for (int rv1 = 0; rv1 < 5; ++rv1) {
      pool_min_local[0] = min(pool_min_local[0], A[(((((((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 9)) / 81) * 401408) + (((((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 8)) % 162) / 9) * 21504)) + (rv0 * 7168)) + ((((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 7)) % 18) * 384)) + (rv1 * 128)) + (((int)threadIdx.x) & 127))]);
    }
  }
  pool_min[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = pool_min_local[0];
}