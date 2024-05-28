
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
    for (size_t i = 0; i < 192; i++) {
        output[i] = signf(input[i]);
    }
}