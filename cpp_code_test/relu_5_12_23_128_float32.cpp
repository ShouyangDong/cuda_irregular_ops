
extern "C" float reluf(float input) {
    return input > 0 ? input : 0;
}
extern "C" void relu_kernel(float* input, float* output) {
    for (size_t i = 0; i < 5; i++) {
        for (size_t j = 0; j < 12; j++) {
            for (size_t k = 0; k < 23; k++) {
                for (size_t l = 0; l < 128; l++) {
                    output[i * 12 * 23 * 128 + j * 23 * 128 + k * 128 + l] = reluf(input[i * 12 * 23 * 128 + j * 23 * 128 + k * 128 + l]);
                }
            }
        }
    }
}