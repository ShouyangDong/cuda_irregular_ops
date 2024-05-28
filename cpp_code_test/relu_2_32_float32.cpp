
extern "C" float reluf(float input) {
    return input > 0 ? input : 0;
}
extern "C" void relu_kernel(float* input, float* output) {
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 32; j++) {
            output[i * 32 + j] = reluf(input[i * 32 + j]);
        }
    }
}