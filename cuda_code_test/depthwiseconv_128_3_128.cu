__global__ void depthwise_convolution(float* input, float* filter, float* output, int input_height, int input_width, int filter_size, int output_height, int output_width) {
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(tid_x < output_width && tid_y < output_height) {
        int output_idx = tid_y * output_width + tid_x;
        
        for(int c = 0; c < input_channels; c++) {
            for(int i = 0; i < filter_size; i++) {
                for(int j = 0; j < filter_size; j++) {
                    int input_idx = (tid_y + i) * input_width + (tid_x + j);
                    int filter_idx = c * filter_size * filter_size + i * filter_size + j;
                    
                    output[output_idx] += input[input_idx] * filter[filter_idx];
                }
            }
        }
    }
}


extern "C" void depthwiseconv_kernel(float *input, float *kernel, float *output, int input_size, int kernel_size, int depth) {
    float *d_input, *d_kernel, *d_output;

    cudaMalloc(&d_input, input_size * depth * sizeof(float));
    cudaMalloc(&d_kernel, kernel_size * depth * sizeof(float));
    cudaMalloc(&d_output, input_size * depth * sizeof(float));

    cudaMemcpy(d_input, input, input_size * depth * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_size * depth * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(128);
    dim3 numBlocks((input_size + blockSize.x - 1) / blockSize.x);

    depthwise_convolution<<<numBlocks, blockSize>>>(d_input, d_kernel, d_output, input_size, kernel_size, depth);

    cudaMemcpy(output, d_output, input_size * depth * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}