

__global__ void cuda_rms_norm(float* A, float* B) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float eps = 1e-5f;

  if (idx < 8192) {
    // Calculate sum
    float sum = 0.0;
    for (int j = 0; j < 4096; j++) {
      sum += A[idx * 4096 + j] * A[idx * 4096 + j];
    }

    // Calculate mean
    float mean = sum / 4096;

    // Calculate scale
    float scale = 1.0 / sqrt(mean + eps);

    // Normalize and store in B
    for (int j = 0; j < 4096; j++) {
      B[idx * 4096 + j] = A[idx * 4096 + j] * scale;
    }
  }
}

extern "C" void rms_norm_kernel(float* A, float* B) {
  // Allocate memory on the device
  float *d_A, *d_B;
  int size_1 = 8192;
  int size_2 = 4096;
  int num_elements = size_1 * size_2;
  cudaMalloc(&d_A, num_elements * sizeof(float));
  cudaMalloc(&d_B, num_elements * sizeof(float));

  // Copy data from host to device
  cudaMemcpy(d_A, A, num_elements * sizeof(float), cudaMemcpyHostToDevice);

  // Define grid and block dimensions
  int block_size = 256;
  int num_blocks = (size + block_size - 1) / block_size;

  // Launch kernel
  cuda_rms_norm<<<num_blocks, block_size>>>(d_A, d_B);

  // Copy the result back to host
  cudaMemcpy(B, d_B, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
  // Free device memory
  cudaFree(d_A);
  cudaFree(d_B);
}
