__global__ void __launch_bounds__(60) sign_kernel(float* __restrict__ A, float* __restrict__ T_sign) {
  T_sign[((int)threadIdx.x)] = ((0.000000e+00f < A[((int)threadIdx.x)]) ? 1.000000e+00f : ((A[((int)threadIdx.x)] < 0.000000e+00f) ? -1.000000e+00f : 0.000000e+00f));
}