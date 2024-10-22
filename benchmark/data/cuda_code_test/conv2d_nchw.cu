__global__ void conv2d_nchw(float* input, float* output, float* kernel, 
                            int N, int C, int H, int W, int K, int R, int S) {
    // N: Batch size
    // C: Input channels
    // H: Input height
    // W: Input width
    // K: Output channels (number of filters)
    // R: Kernel height
    // S: Kernel width
    
    int n = blockIdx.z;  // Batch index
    int k = blockIdx.y;  // Output channel (filter) index
    int h_out = blockIdx.x * blockDim.y + threadIdx.y;  // Output height index
    int w_out = threadIdx.x;  // Output width index

    // Calculate output height and width
    int H_out = H;
    int W_out = W;

    if (h_out < H_out && w_out < W_out) {
        float output_value = 0.0f;

        // Iterate over input channels (C) and kernel dimensions (R, S)
        for (int c = 0; c < C; ++c) {
            for (int r = 0; r < R; ++r) {
                for (int s = 0; s < S; ++s) {
                    // Calculate input pixel indices
                    int h_in = h_out + r - R / 2;
                    int w_in = w_out + s - S / 2;

                    // Ensure input indices are within bounds
                    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                        // Index for input, kernel, and output
                        int input_index = n * (C * H * W) + c * (H * W) + h_in * W + w_in;
                        int kernel_index = k * (C * R * S) + c * (R * S) + r * S + s;

                        output_value += input[input_index] * kernel[kernel_index];
                    }
                }
            }
        }

        // Index for output
        int output_index = n * (K * H_out * W_out) + k * (H_out * W_out) + h_out * W_out + w_out;
        output[output_index] = output_value;
    }
}

extern "C" void conv2d_nchw_kernel(float *output, float *input, float *kernel, 
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

    conv2d_nchw<<<numBlocks, blockSize>>>(d_input, d_kernel, d_output);

    cudaMemcpy(output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}
