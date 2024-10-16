
float sigmoidf(float input) {
    return 1 / (1 + exp(-1 * input));
}
extern "C" void sigmoid_kernel(float *output, float *input) {
    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j < 10; j++) {
            for (size_t k = 0; k < 64; k++) {
                output[i * 10 * 64 + j * 64 + k] = sigmoidf(input[i * 10 * 64 + j * 64 + k]);
            }
        }
    }
}