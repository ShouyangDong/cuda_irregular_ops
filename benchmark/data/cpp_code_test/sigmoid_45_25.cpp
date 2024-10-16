
float sigmoidf(float input) {
    return 1 / (1 + exp(-1 * input));
}
extern "C" void sigmoid_kernel(float *output, float *input) {
    for (size_t i = 0; i < 45; i++) {
        for (size_t j = 0; j < 25; j++) {
            output[i * 25 + j] = sigmoidf(input[i * 25 + j]);
        }
    }
}