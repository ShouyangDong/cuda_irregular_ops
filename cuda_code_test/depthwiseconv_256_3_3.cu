__global__ void depthwise_convolution(float* input, float* filter, float* output) {
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(tid_x < 254 && tid_y < 254) {
        int output_idx = tid_y * 254 + tid_x;
        
        for(int c = 0; c < 3; c++) {
            for(int i = 0; i < 3; i++) {
                for(int j = 0; j < 3; j++) {
                    int input_idx = (tid_y + i) * 256 + (tid_x + j);
                    int filter_idx = c * 3 * 3 + i * 3 + j;
                    
                    output[output_idx] += input[input_idx] * filter[filter_idx];
                }
            }
        }
    }
}


extern "C" void depthwiseconv_kernel(float* input, float* filter, float* output, int input_height, int kernel_size, int input_channels) {
    float *d_input, *d_kernel, *d_output;
    int input_size = input_height * input_height * input_channels;
    int output_height = input_height - kernel_size + 1;
    int output_width = input_height - kernel_size + 1;
    int kernel_size = kernel_size * kernel_size * input_channels;
    int output_size = output_height * output_width * input_channels;
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_kernel, kernel_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));

    cudaMemcpy(d_input, input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(128);
    dim3 numBlocks((input_size + blockSize.x - 1) / blockSize.x);

    depthwise_convolution<<<numBlocks, blockSize>>>(d_input, d_kernel, d_output);

    cudaMemcpy(output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}
