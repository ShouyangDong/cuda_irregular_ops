__global__ void __launch_bounds__(36) softmax_kernel(float* __restrict__ A, float* __restrict__ T_softmax_exp) {
  if (threadIdx.x < 36) {
      
      float maxVal = A[threadIdx.x * 128];
      for (int i = 1; i < 128; ++i) {
          if (A[threadIdx.x * 128 + i] > maxVal) {
              maxVal = A[threadIdx.x * 128 + i];
          }
      }
      
      
      float denom = 0.0f;
      for (int i = 0; i < 128; ++i) {
          T_softmax_exp[threadIdx.x * 128 + i] = expf(A[threadIdx.x * 128 + i] - maxVal);
          denom += T_softmax_exp[threadIdx.x * 128 + i];
      }
      
      
      for (int i = 0; i < 128; ++i) {
          T_softmax_exp[threadIdx.x * 128 + i] /= denom;
      }
  }
}
