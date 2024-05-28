
extern "C" float sigmoidf(float input) {
    return 1 / (1 + exp(-1 * input));
}
extern "C" void sigmoid_kernel(float* input, float* output) {
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 4; j++) {
            for (size_t k = 0; k < 5; k++) {
                output[i * 4 * 5 + j * 5 + k] = sigmoidf(input[i * 4 * 5 + j * 5 + k]);
            }
        }
    }
}