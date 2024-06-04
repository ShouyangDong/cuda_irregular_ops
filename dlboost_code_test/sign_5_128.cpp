
float signf(float input){
    if (input > 0) {
        return 1;
    } else if (input < 0) {
        return -1;
    } else {
        return 0;
    }
}
extern "C" void sign_kernel(float* input, float* output) {
    for (size_t i = 0; i < 5; i++) {
        for (size_t j = 0; j < 128; j++) {
            output[i * 128 + j] = signf(input[i * 128 + j]);
        }
    }
}