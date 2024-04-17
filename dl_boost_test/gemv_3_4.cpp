extern "C" void gemv_kernel(float *y, float *A, float *x) {
    for (int i = 0; i < 3; i++) {
        y[i] = 0;
        for (int j = 0; j < 4; j++) {
            y[i] += A[i * 4 + j] * x[j];
        }
    }
}