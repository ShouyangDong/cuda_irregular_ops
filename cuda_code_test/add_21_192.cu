extern "C" __global__ void __launch_bounds__(1024) add_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ T_add) {
  if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 4032) {
    T_add[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (A[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] + B[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
  }
}