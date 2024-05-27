extern "C" __global__ void __launch_bounds__(1024) minpool_kernel(float* __restrict__ A, float* __restrict__ pool_min) {
  float pool_min_local[1];
  pool_min_local[0] = 3.402823e+38f;
  for (int rv0 = 0; rv0 < 5; ++rv0) {
    for (int rv1 = 0; rv1 < 5; ++rv1) {
      if (((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 8)) < 125) {
        pool_min_local[0] = min(pool_min_local[0], A[(((((((((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 8)) / 25) * 65536) + (((((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 7)) % 50) / 5) * 6144)) + (rv0 * 2048)) + ((((((int)blockIdx.x) * 6) + (((int)threadIdx.x) >> 6)) % 10) * 192)) + (rv1 * 64)) + (((int)threadIdx.x) & 63))]);
      }
    }
  }
  if (((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 8)) < 125) {
    pool_min[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = pool_min_local[0];
  }
}