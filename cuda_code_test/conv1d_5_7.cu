__global__ void conv1d(float *input, float *kernel, float *output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int output_idx = idx + 3 / 2;

    if (idx < 7) {
        float sum = 0.0f;
        for (int i = 0; i < 3; i++) {
            int input_idx = idx + i - 3/2;
            if (input_idx >= 0 && input_idx < 7) {
                sum += input[input_idx] * kernel[i];
            }
        }
        output[output_idx] = sum;
    }
}


extern "C" void conv1d_kernel(float *output, float *input, float *kernel, int input_size, int output_size) {
    float *d_input, *d_kernel, *d_output;
    int kernel_size = input_size - output_size;
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_kernel, kernel_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));

    cudaMemcpy(d_input, input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(7);
    dim3 numBlocks((input_size + blockSize.x - 1) / blockSize.x);

    conv1d<<<numBlocks, blockSize>>>(d_input, d_kernel, d_output);

    cudaMemcpy(output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}
