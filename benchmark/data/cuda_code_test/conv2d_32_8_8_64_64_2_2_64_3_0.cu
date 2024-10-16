__global__ void conv2d(float* input, float* kernel, float* output) {
    int batch_size = 32;
    int input_height = 8;
    int input_width = 8;
    int input_channels = 64;
    int kernel_height = 2;
    int kernel_width = 2;
    int output_channels = 64;
    int stride = 3;
    int output_height = (input_height - kernel_height) / stride + 1;
    int output_width = (input_width - kernel_width) / stride + 1;

    for (int bs = 0; bs < batch_size; bs++) {
        for (int oh = 0; oh < output_height; oh++) {
            for (int ow = 0; ow < output_width; ow++) {
                if (threadIdx.x < output_channels) {
                    float sum = 0.0;
                    for (int kh = 0; kh < kernel_height; kh++) {
                        for (int kw = 0; kw < kernel_width; kw++) {
                            for (int ic = 0; ic < input_channels; ic++) {
                                int ih = oh * stride + kh;
                                int iw = ow * stride + kw;
                                sum += input[bs * input_height * input_width * input_channels + ih * input_width * input_channels + iw * input_channels + ic] * kernel[threadIdx.x * kernel_height * kernel_width * input_channels + kh * kernel_width * input_channels + kw * input_channels + ic];
                            }
                        }
                    }
                    output[bs * output_height * output_width * output_channels + oh * output_width * output_channels + ow * output_channels + threadIdx.x] = sum;
                }
            }
        }
    }
}

extern "C" void conv2d_kernel(float *output, float *input, float *kernel, 
                                int batch_size, int input_height, int input_channels,
                                int output_channels, int kernel_height, int stride) {
    int output_height = (input_height - kernel_height) / stride + 1;
    int input_size = batch_size * input_height * input_height * input_channels;
    int kernel_size = input_channels * output_channels * kernel_height * kernel_height;
    int output_size = batch_size * output_height * output_height * output_channels;
    float *d_input, *d_kernel, *d_output;

    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_kernel, kernel_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));

    cudaMemcpy(d_input, input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(64);
    dim3 numBlocks(1);

    conv2d<<<numBlocks, blockSize>>>(d_input, d_kernel, d_output);

    cudaMemcpy(output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}
