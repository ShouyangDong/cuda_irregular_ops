extern "C" void depthwiseconv_kernel(float *output, float *input, float *kernel){
    int in_depth = 3;
    int input_height = 6;
    int input_width = 6;
    int kernel_size = 3;
    int output_height = input_height - kernel_size + 1;
    int output_width = input_width - kernel_size + 1;

    for (int c = 0; c < in_depth; ++c) {
        for (int i = 0; i < output_height; ++i) {
            for (int j = 0; j < output_width; ++j) {
                output[i * output_width *  in_depth + j * in_depth + c] = 0.0;
                for (int fi = 0; fi < kernel_size; ++fi) {
                    for (int fj = 0; fj < kernel_size; ++fj) {
                        output[i * output_width *  in_depth + j * in_depth + c] += input[(i + fi)* in_depth * input_width + (j + fj) * in_depth + c] * kernel[fi * kernel_size * kernel_size + fj * kernel_size + c];;
                    }
                }
            }
        }
    }
}
