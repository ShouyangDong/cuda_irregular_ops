
extern "C" float sigmoidf(float input) {
    return 1 / (1 + exp(-1 * input));
}
extern "C" void sigmoid_kernel(float* input, float* output) {
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 32; j++) {
            output[i * 32 + j] = sigmoidf(input[i * 32 + j]);
        }
    }
}