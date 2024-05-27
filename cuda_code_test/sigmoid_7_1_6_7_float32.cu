extern "C" __global__ void __launch_bounds__(294) sigmoid_kernel(float* __restrict__ A, float* __restrict__ compute) {
  compute[((int)threadIdx.x)] = (1.000000e+00f / (1.000000e+00f + __expf((0.000000e+00f - A[((int)threadIdx.x)]))));
}