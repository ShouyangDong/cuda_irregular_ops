
extern "C" float sigmoidf(float input) {
    return 1 / (1 + exp(-1 * input));
}
extern "C" void sigmoid_kernel(float* input, float* output) {
    for (size_t i = 0; i < 5; i++) {
        for (size_t j = 0; j < 128; j++) {
            output[i * 128 + j] = sigmoidf(input[i * 128 + j]);
        }
    }
}