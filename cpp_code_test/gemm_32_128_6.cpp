extern "C" void gemm_kernel(float *result, float *A, float *B) {
    for (int j = 0; j < 32; j++) {
        for (int k = 0; k < 6; k++) {
            result[j * 6 + k] = 0;
            for (int l = 0; l < 128; l++) {
                result[j * 6 + k] += A[j *128 + l] * B[l * 6 + k];
            }
        }
    }
}
