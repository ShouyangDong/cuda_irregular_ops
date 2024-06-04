
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
    for (size_t i = 0; i < 45; i++) {
        for (size_t j = 0; j < 25; j++) {
            output[i * 25 + j] = signf(input[i * 25 + j]);
        }
    }
}