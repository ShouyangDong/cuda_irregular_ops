__global__ void cuda_layer_norm(float* A, float* gamma, float* beta, float* B) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < 8) {
    float mean = 0.0;
    float variance = 0.0;
    float diff[8];
    // Calculate mean
    for (int i_mean = 0; i_mean < 8; i_mean++) {
      mean += A[idx * 8 + i_mean];
    }
    mean /= 8;
    // Calculate variance
    for (int i_diff = 0; i_diff < 8; i_diff++) {
      diff[i_diff] = A[idx * 8 + i_diff] - mean;
    }

    for (int i_pow = 0; i_pow < 8; i_pow++) {
      diff[i_pow] = diff[i_pow] * diff[i_pow];
    }
    for (int i_var = 0; i_var < 8; i_var++) {
      variance += diff[i_var];
    }
    variance = sqrt(variance / 8);

    // Normalize A
    for (int i_norm = 0; i_norm < 8; i_norm++) {
      diff[i_norm] = (A[idx * 8 + i_norm] - mean);
    }

    for (int i_mul = 0; i_mul < 8; i_mul++) {
      diff[i_mul] = diff[i_mul] * gamma[i_mul];
    }

    for (int i_div = 0; i_div < 8; i_div++) {
      diff[i_div] = diff[i_div] / (variance + 1e-5f);
    }

    for (int i_bet = 0; i_bet < 8; i_bet++) {
      B[idx * 8 + i_bet] = diff[i_bet] + beta[i_bet];
    }
  }
}

extern "C" void layer_norm_kernel(float* A, float* gamma, float* beta,
                                  float* B) {
  // Allocate memory on the device
  float *d_A, *d_B, *d_gamma, *d_beta;
  int batch_size = 2;
  int seq_length = 4;
  int d_model = 8;
  int num_elements = batch_size * seq_length * d_model;
  cudaMalloc(&d_A, num_elements * sizeof(float));
  cudaMalloc(&d_B, num_elements * sizeof(float));
  cudaMalloc(&d_gamma, d_model * sizeof(float));
  cudaMalloc(&d_beta, d_model * sizeof(float));

  // Copy data from host to device
  cudaMemcpy(d_A, A, num_elements * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_gamma, gamma, d_model * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_beta, beta, d_model * sizeof(float), cudaMemcpyHostToDevice);

  // Define grid and block dimensions
  int block_size = 256;
  int num_blocks = (batch_size * seq_length + block_size - 1) / block_size;

  // Launch kernel
  cuda_layer_norm<<<num_blocks, block_size>>>(d_A, d_gamma, d_beta, d_B);

  // Copy the result back to host
  cudaMemcpy(B, d_B, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
  // Free device memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_gamma);
  cudaFree(d_beta);
}
