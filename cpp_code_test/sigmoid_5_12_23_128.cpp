
float sigmoidf(float input) {
    return 1 / (1 + exp(-1 * input));
}
extern "C" void sigmoid_kernel(float* input, float* output) {
    for (size_t i = 0; i < 5; i++) {
        for (size_t j = 0; j < 12; j++) {
            for (size_t k = 0; k < 23; k++) {
                for (size_t l = 0; l < 128; l++) {
                    output[i * 12 * 23 * 128 + j * 23 * 128 + k * 128 + l] = sigmoidf(input[i * 12 * 23 * 128 + j * 23 * 128 + k * 128 + l]);
                }
            }
        }
    }
}