
float geluf(float x) {
    return 0.5 * x * (1 + tanh(sqrt(2 / M_PI) * (x + 0.044715 * pow(x, 3))));
}
extern "C" void gelu_kernel(float *input, float *output) {
    for (int i = 0; i < 192; i++) {
        output[i] = geluf(input[i]);
    }
}