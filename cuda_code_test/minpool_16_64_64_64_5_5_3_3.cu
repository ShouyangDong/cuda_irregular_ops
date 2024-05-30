__global__ void __launch_bounds__(1024) minpool_kernel(float* __restrict__ A, float* __restrict__ pool_min) {
  float pool_min_local[1];
  pool_min_local[0] = 3.402823e+38f;
  for (int rv0 = 0; rv0 < 5; ++rv0) {
    for (int rv1 = 0; rv1 < 5; ++rv1) {
      pool_min_local[0] = min(pool_min_local[0], A[(((((((((int)blockIdx.x) / 25) * 262144) + (((((((int)blockIdx.x) % 25) * 4) + (((int)threadIdx.x) >> 8)) / 5) * 12288)) + (rv0 * 4096)) + ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 6)) % 20) * 192)) + (rv1 * 64)) + (((int)threadIdx.x) & 63))]);
    }
  }
  pool_min[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = pool_min_local[0];
}