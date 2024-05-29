extern "C" __global__ void __launch_bounds__(12) softmax_kernel(float* __restrict__ A, float* __restrict__ T_softmax_maxelem) {
  if (threadIdx.x < 12) {
      
      float maxVal = A[threadIdx.x * 5];
      for (int i = 1; i < 5; ++i) {
          if (A[threadIdx.x * 5 + i] > maxVal) {
              maxVal = A[threadIdx.x * 5 + i];
          }
      }
      
      
      float denom = 0.0f;
      for (int i = 0; i < 5; ++i) {
          T_softmax_maxelem[threadIdx.x * 5 + i] = expf(A[threadIdx.x * 5 + i] - maxVal);
          denom += T_softmax_maxelem[threadIdx.x * 5 + i];
      }
      
      
      for (int i = 0; i < 5; ++i) {
          T_softmax_maxelem[threadIdx.x * 5 + i] /= denom;
      }
  }
}

