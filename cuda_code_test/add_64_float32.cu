extern "C" __global__ void __launch_bounds__(64) add_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ T_add) {
  T_add[((int)threadIdx.x)] = (A[((int)threadIdx.x)] + B[((int)threadIdx.x)]);
}