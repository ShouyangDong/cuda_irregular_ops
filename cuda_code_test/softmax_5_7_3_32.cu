extern "C" __global__ void __launch_bounds__(105) softmax_kernel(float* __restrict__ A, float* __restrict__ T_softmax_maxelem) {
  if (threadIdx.x < 105) {
      
      float maxVal = A[threadIdx.x * 32];
      for (int i = 1; i < 32; ++i) {
          if (A[threadIdx.x * 32 + i] > maxVal) {
              maxVal = A[threadIdx.x * 32 + i];
          }
      }
      
      
      float denom = 0.0f;
      for (int i = 0; i < 32; ++i) {
          T_softmax_maxelem[threadIdx.x * 32 + i] = expf(A[threadIdx.x * 32 + i] - maxVal);
          denom += T_softmax_maxelem[threadIdx.x * 32 + i];
      }
      
      
      for (int i = 0; i < 32; ++i) {
          T_softmax_maxelem[threadIdx.x * 32 + i] /= denom;
      }
  }
}
