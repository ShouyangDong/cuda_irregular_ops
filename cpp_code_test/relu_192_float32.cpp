
extern "C" float reluf(float input) {
    return input > 0 ? input : 0;
}
extern "C" void relu_kernel(float* input, float* output) {
    for (size_t i = 0; i < 192; i++) {
        output[i] = reluf(input[i]);
    }
}