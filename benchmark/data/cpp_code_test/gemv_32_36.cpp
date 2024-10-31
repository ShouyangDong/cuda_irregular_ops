extern "C" void gemv_kernel(float *y, float *A, float *x) {
  for (int i = 0; i < 32; i++) {
    y[i] = 0;
    for (int j = 0; j < 36; j++) {
      y[i] += A[i * 36 + j] * x[j];
    }
  }
}