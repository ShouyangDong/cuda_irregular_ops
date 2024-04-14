#include <iostream>

void depthwise_convolution(float *output, float *input, float *kernel){
    int input_channels = 3;
    int input_height = 5;
    int input_width = 5;
    int kernel_size = 3;
    int output_height = input_height - kernel_size + 1;
    int output_width = input_width - kernel_size + 1;

    for (int c = 0; c < input_channels; c++) {
        for (int i = 0; i < output_height; i++) {
            for (int j = 0; j < output_width; j++) {
                output[c*output_height*output_width + i*output_width + j] = 0;
                for (int m = 0; m < kernel_size; m++) {
                    for (int n = 0; n < kernel_size; n++) {
                        output[c*output_height*output_width + i*output_width + j] += input[c*input_height*input_width + (i+m)*input_width + (j+n)] * kernel[m*kernel_size + n];
                    }
                }
            }
        }
    }
}
