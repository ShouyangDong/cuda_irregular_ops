extern "C" void softmax_kernel(float *input, float *output) {
    float max_val = input[0];
    for (int i = 1; i < 192; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    float sum = 0.0;
    for (int i = 0; i < 192; i++) {
        output[i] = exp(input[i] - max_val);
        sum += output[i];
    }

    for (int i = 0; i < 192; i++) {
        output[i] /= sum;
    }
}