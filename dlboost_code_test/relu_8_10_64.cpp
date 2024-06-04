
float reluf(float input) {
    return input > 0 ? input : 0;
}
extern "C" void relu_kernel(float *output, float *input) {
    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 10; j++) {
            for (size_t k = 0; k < 64; k++) {
                output[i * 10 * 64 + j * 64 + k] = reluf(input[i * 10 * 64 + j * 64 + k]);
            }
        }
    }
}