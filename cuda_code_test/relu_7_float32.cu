extern "C" __global__ void __launch_bounds__(7) relu_kernel(float* __restrict__ A, float* __restrict__ compute) {
  compute[((int)threadIdx.x)] = max(A[((int)threadIdx.x)], 0.000000e+00f);
}