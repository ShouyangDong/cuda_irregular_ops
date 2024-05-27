extern "C" __global__ void __launch_bounds__(1) softmax_kernel(float* __restrict__ A, float* __restrict__ T_softmax_maxelem) {
  if (threadIdx.x < 1) {
      
      float maxVal = A[0];
      for (int i = 1; i < 192; ++i) {
          if (A[i] > maxVal) {
              maxVal = A[i];
          }
      }
      
      float denom = 0.0f;
      for (int i = 0; i < 192; ++i) {
          T_softmax_maxelem[i] = expf(A[i] - maxVal);
          denom += T_softmax_maxelem[i];
      }
      
      
      for (int i = 0; i < 192; ++i) {
          T_softmax_maxelem[i] /= denom;
      }
  }
}
