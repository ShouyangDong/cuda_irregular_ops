extern "C" void softmax_kernel(float *x, float *output) {

    float max_val = -INFINITY;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 32; j++) {
            if (x[i * 32 + j] > max_val) {
                max_val = x[i * 32 + j];
            }
        }

        float sum_exp = 0.0;
        for (int j = 0; j < 32; j++) {
            int index = i * 32 + j;
            float exp_val = expf(x[index] - max_val);
            output[index] = exp_val;
            sum_exp += exp_val;
        }

        for (int j = 0; j < 32; j++) {
            output[i * 32 + j] /= sum_exp;
        }
    }
}