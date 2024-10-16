
float reluf(float input) {
    return input > 0 ? input : 0;
}
extern "C" void relu_kernel(float *output, float *input) {
    for (size_t i = 0; i < 45; i++) {
        for (size_t j = 0; j < 25; j++) {
            output[i * 25 + j] = reluf(input[i * 25 + j]);
        }
    }
}