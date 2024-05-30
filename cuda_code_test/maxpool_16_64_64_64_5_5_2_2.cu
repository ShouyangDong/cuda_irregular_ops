__global__ void __launch_bounds__(1024) maxpool_kernel(float* __restrict__ A, float* __restrict__ pool_max) {
  float pool_max_local[1];
  pool_max_local[0] = -3.402823e+38f;
  for (int rv0 = 0; rv0 < 5; ++rv0) {
    for (int rv1 = 0; rv1 < 5; ++rv1) {
      pool_max_local[0] = max(pool_max_local[0], A[(((((((((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 8)) / 225) * 262144) + (((((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 7)) % 450) / 15) * 8192)) + (rv0 * 4096)) + ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 6)) % 30) * 128)) + (rv1 * 64)) + (((int)threadIdx.x) & 63))]);
    }
  }
  pool_max[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = pool_max_local[0];
}