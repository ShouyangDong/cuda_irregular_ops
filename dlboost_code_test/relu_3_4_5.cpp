
extern "C" float reluf(float input) {
    return input > 0 ? input : 0;
}
extern "C" void relu_kernel(float* input, float* output) {
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 4; j++) {
            for (size_t k = 0; k < 5; k++) {
                output[i * 4 * 5 + j * 5 + k] = reluf(input[i * 4 * 5 + j * 5 + k]);
            }
        }
    }
}