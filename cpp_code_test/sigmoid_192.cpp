
float sigmoidf(float input) {
    return 1 / (1 + exp(-1 * input));
}
extern "C" void sigmoid_kernel(float *output, float *input) {
    for (size_t i = 0; i < 192; i++) {
        output[i] = sigmoidf(input[i]);
    }
}