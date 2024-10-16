extern "C" void gemv_kernel(float *y, float *A, float *x) {
    for (int i = 0; i < 125; i++) {
        y[i] = 0;
        for (int j = 0; j < 128; j++) {
            y[i] += A[i * 128 + j] * x[j];
        }
    }
}