extern "C" __global__ void __launch_bounds__(1024) relu_kernel(float* __restrict__ A, float* __restrict__ compute) {
  if (((blockIdx.x * 1024) + (threadIdx.x)) < 5120) {
  compute[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = max(A[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))], 0.000000e+00f);
}
}
