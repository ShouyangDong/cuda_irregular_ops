#include <stdio.h>

extern "C" void conv_1d_kernel(float *output, float *input, float *kernel) {
    for (int i = 0; i < 190; i++) {
        output[i] = 0;
        for (int j = 0; j < 3; j++) {
            output[i] += input[i + j] * kernel[j];
        }
    }
}
