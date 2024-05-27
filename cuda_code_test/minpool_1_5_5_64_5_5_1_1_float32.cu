extern "C" __global__ void __launch_bounds__(1024) minpool_kernel(float* __restrict__ A, float* __restrict__ pool_min) {
  float pool_min_local[1];
  pool_min_local[0] = 3.402823e+38f;
  for (int rv0 = 0; rv0 < 5; ++rv0) {
    for (int rv1 = 0; rv1 < 5; ++rv1) {
      if (((int)threadIdx.x) < 64) {
        pool_min_local[0] = min(pool_min_local[0], A[(((rv0 * 320) + (rv1 * 64)) + ((int)threadIdx.x))]);
      }
    }
  }
  if (((int)threadIdx.x) < 64) {
    pool_min[((int)threadIdx.x)] = pool_min_local[0];
  }
}