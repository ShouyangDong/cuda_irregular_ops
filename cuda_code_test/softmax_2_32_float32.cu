extern "C" __global__ void __launch_bounds__(2) softmax_kernel(float* __restrict__ A, float* __restrict__ T_softmax_norm) {
  if (threadIdx.x < 2) {
    int rowStart = threadIdx.x * 32;
    
    float maxVal = A[rowStart];
    for (int i = 1; i < 32; ++i) {
        if (A[rowStart + i] > maxVal) {
            maxVal = A[rowStart + i];
        }
    }
    
    float denom = 0.0f;
    for (int i = 0; i < 32; ++i) {
        T_softmax_norm[rowStart + i] = expf(A[rowStart + i] - maxVal);
        denom += T_softmax_norm[rowStart + i];
    }
    
    for (int i = 0; i < 32; ++i) {
        T_softmax_norm[rowStart + i] /= denom;
    }
  }
}
