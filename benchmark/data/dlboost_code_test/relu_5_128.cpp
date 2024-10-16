
float reluf(float input) {
    return input > 0 ? input : 0;
}
extern "C" void relu_kernel(float *output, float *input) {
    for (size_t i = 0; i < 5; i++) {
        for (size_t j = 0; j < 128; j++) {
            output[i * 128 + j] = reluf(input[i * 128 + j]);
        }
    }
}