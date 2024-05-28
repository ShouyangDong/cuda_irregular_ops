
extern "C" float signf(float input){
    if (input > 0) {
        return 1;
    } else if (input < 0) {
        return -1;
    } else {
        return 0;
    }
}
extern "C" void sign_kernel(float* input, float* output) {
    for (size_t i = 0; i < 12; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 128; k++) {
                output[i * 3 * 128 + j * 128 + k] = signf(input[i * 3 * 128 + j * 128 + k]);
            }
        }
    }
}