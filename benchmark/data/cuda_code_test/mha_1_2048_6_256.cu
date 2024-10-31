__global__ void multi_head_attention(const float *__restrict__ queries,
                                     const float *__restrict__ keys,
                                     const float *__restrict__ values,
                                     float *__restrict__ output, int seq_len,
                                     int num_heads, int head_dim) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < seq_len * num_heads * head_dim) {
    int head_id = (tid / head_dim) % num_heads;
    int seq_id = tid / (num_heads * head_dim);
    int dim_id = tid % head_dim;

    // Perform scaled dot-product attention
    float attn_score = 0.0f;
    for (int i = 0; i < head_dim; ++i) {
      attn_score +=
          queries[seq_id * num_heads * head_dim + head_id * head_dim + i] *
          keys[seq_id * num_heads * head_dim + head_id * head_dim + dim_id];
    }

    attn_score /= sqrtf((float)head_dim);

    // Softmax over attention scores is skipped in this simple example for
    // simplicity.

    // Compute weighted values
    for (int i = 0; i < head_dim; ++i) {
      output[seq_id * num_heads * head_dim + head_id * head_dim + i] +=
          attn_score *
          values[seq_id * num_heads * head_dim + head_id * head_dim + i];
    }
  }
}

extern "C" void mha_kernel(const float *queries, const float *keys,
                           const float *values, float *output, int batch_size,
                           int seq_len, int num_heads, int head_dim) {
  int size = batch_size * seq_len * num_heads * head_dim;
  float *d_queries, *d_keys, *d_values, *d_output;
  cudaMalloc(&d_queries, size * sizeof(float));
  cudaMalloc(&d_keys, size * sizeof(float));
  cudaMalloc(&d_values, size * sizeof(float));
  cudaMalloc(&d_output, size * sizeof(float));
  cudaMemcpy(&d_queries, queries, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(&d_keys, keys, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(&d_values, values, size * sizeof(float), cudaMemcpyHostToDevice);

  int blockSize = 256;
  int gridSize = (seq_len * num_heads * head_dim + blockSize - 1) / blockSize;

  multi_head_attention<<<gridSize, blockSize>>>(
      d_queries, d_keys, d_values, d_output, seq_len, num_heads, head_dim);

  cudaMemcpy(output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_queries);
  cudaFree(d_keys);
  cudaFree(d_values);
  cudaFree(d_output);
}
