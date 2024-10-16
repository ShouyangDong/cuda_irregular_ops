
float signf(float input){
    if (input > 0) {
        return 1;
    } else if (input < 0) {
        return -1;
    } else {
        return 0;
    }
}
extern "C" void sign_kernel(float *output, float *input) {
    for (size_t i = 0; i < 5; i++) {
        for (size_t j = 0; j < 7; j++) {
            for (size_t k = 0; k < 3; k++) {
                for (size_t l = 0; l < 32; l++) {
                    output[i * 7 * 3 * 32 + j * 3 * 32 + k * 32 + l] = signf(input[i * 7 * 3 * 32 + j * 3 * 32 + k * 32 + l]);
                }
            }
        }
    }
}