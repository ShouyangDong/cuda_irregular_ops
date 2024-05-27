extern "C" __global__ void __launch_bounds__(1024) sign_kernel(float* __restrict__ A, float* __restrict__ T_sign) {
  if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 176640) {
    T_sign[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = ((0.000000e+00f < A[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]) ? 1.000000e+00f : ((A[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] < 0.000000e+00f) ? -1.000000e+00f : 0.000000e+00f));
  }
}