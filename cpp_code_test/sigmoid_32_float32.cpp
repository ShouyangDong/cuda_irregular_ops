
extern "C" float sigmoidf(float input) {
    return 1 / (1 + exp(-1 * input));
}
extern "C" void sigmoid_kernel(float* input, float* output) {
    for (size_t i = 0; i < 32; i++) {
        output[i] = sigmoidf(input[i]);
    }
}