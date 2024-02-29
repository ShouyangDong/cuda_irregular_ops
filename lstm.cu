extern "C" __global__ void __launch_bounds__(192) default_function_kernel(float* __restrict__ Wh2h, float* __restrict__ Xi2h, float* __restrict__ lstm_scan, float* __restrict__ lstm_scan_1, int num_step, int stride, int stride_1, int stride_2, int stride_3, int stride_4, int stride_5, int stride_6) {
  float Wh2h_local[288];
  __shared__ float placeholder_d_shared[24];
  float s_h2h[4];
  __shared__ float placeholder_shared[576];
  float s_h2h_rf[1];
  __shared__ float red_buf0[192];
  float next_c[1];
  float next_h[1];
  for (int i = 0; i < 72; i++) {
    Wh2h_local[0 + i] = Wh2h[((((((int)blockIdx.x) * 13824) + (((int)threadIdx.x) * 576)) + (((int)threadIdx.y) * 72)) + i)];
  }

  for (int i = 0; i < 72; i++) {
    Wh2h_local[72 + i] = Wh2h[((((((int)blockIdx.x) * 13824) + (((int)threadIdx.x) * 576)) + (((int)threadIdx.y) * 72)) + 331776 + i)];
  }

  for (int i = 0; i < 72; i++) {
    Wh2h_local[144 + i] = Wh2h[((((((int)blockIdx.x) * 13824) + (((int)threadIdx.x) * 576)) + (((int)threadIdx.y) * 72)) + 663552 + i)];
  }

  for (int i = 0; i < 72; i++) {
    Wh2h_local[216 + i] = Wh2h[((((((int)blockIdx.x) * 13824) + (((int)threadIdx.x) * 576)) + (((int)threadIdx.y) * 72)) + 995328 + i)];
  }
  lstm_scan[(((((int)blockIdx.x) * 24) + ((int)threadIdx.x)) * stride_1)] = 0.000000e+00f;
  lstm_scan_1[(((((int)blockIdx.x) * 24) + ((int)threadIdx.x)) * stride_3)] = 0.000000e+00f;
  for (int lstm_scan_idx = 0; lstm_scan_idx < (num_step - 1); ++lstm_scan_idx) {
    __syncthreads();
    for (int ax2 = 0; ax2 < 24; ++ax2) {
      placeholder_d_shared[ax2] = lstm_scan_1[((lstm_scan_idx * stride_2) + (((((int)blockIdx.x) * 24) + ax2) * stride_3))];
    }
    for (int ax2_1 = 0; ax2_1 < 4; ++ax2_1) {
      __syncthreads();
      for (int ax2_outer = 0; ax2_outer < 3; ++ax2_outer) {
        placeholder_shared[(((ax2_outer * 192) + (((int)threadIdx.y) * 24)) + ((int)threadIdx.x))] = lstm_scan[((lstm_scan_idx * stride) + ((((ax2_outer * 192) + (((int)threadIdx.y) * 24)) + ((int)threadIdx.x)) * stride_1))];
      }
      s_h2h_rf[0] = 0.000000e+00f;
      __syncthreads();
      for (int ki2h_inner = 0; ki2h_inner < 72; ++ki2h_inner) {
        s_h2h_rf[0] = (s_h2h_rf[0] + (placeholder_shared[((((int)threadIdx.y) * 72) + ki2h_inner)] * Wh2h_local[((ax2_1 * 72) + ki2h_inner)]));
      }
      __syncthreads();
      ((volatile float*)red_buf0)[((((int)threadIdx.x) * 8) + ((int)threadIdx.y))] = s_h2h_rf[0];
      __syncthreads();
      if (((int)threadIdx.y) < 4) {
        ((volatile float*)red_buf0)[((((int)threadIdx.x) * 8) + ((int)threadIdx.y))] = (((volatile float*)red_buf0)[((((int)threadIdx.x) * 8) + ((int)threadIdx.y))] + ((volatile float*)red_buf0)[(((((int)threadIdx.x) * 8) + ((int)threadIdx.y)) + 4)]);
      }
      __syncthreads();
      if (((int)threadIdx.y) < 2) {
        ((volatile float*)red_buf0)[((((int)threadIdx.x) * 8) + ((int)threadIdx.y))] = (((volatile float*)red_buf0)[((((int)threadIdx.x) * 8) + ((int)threadIdx.y))] + ((volatile float*)red_buf0)[(((((int)threadIdx.x) * 8) + ((int)threadIdx.y)) + 2)]);
      }
      __syncthreads();
      if (((int)threadIdx.y) < 1) {
        ((volatile float*)red_buf0)[((((int)threadIdx.x) * 8) + ((int)threadIdx.y))] = (((volatile float*)red_buf0)[((((int)threadIdx.x) * 8) + ((int)threadIdx.y))] + ((volatile float*)red_buf0)[(((((int)threadIdx.x) * 8) + ((int)threadIdx.y)) + 1)]);
      }
      __syncthreads();
      if (((int)threadIdx.y) == 0) {
        s_h2h[ax2_1] = ((volatile float*)red_buf0)[(((int)threadIdx.x) * 8)];
      }
    }
    if (((int)threadIdx.y) == 0) {
      next_c[0] = (((1.000000e+00f / (1.000000e+00f + __expf((0.000000e+00f - (Xi2h[(((stride_5 * 2) + ((lstm_scan_idx + 1) * stride_4)) + (((((int)blockIdx.x) * 24) + ((int)threadIdx.x)) * stride_6))] + s_h2h[2]))))) * placeholder_d_shared[((int)threadIdx.x)]) + ((1.000000e+00f / (1.000000e+00f + __expf((0.000000e+00f - (Xi2h[(((lstm_scan_idx + 1) * stride_4) + (((((int)blockIdx.x) * 24) + ((int)threadIdx.x)) * stride_6))] + s_h2h[0]))))) * tanhf((Xi2h[((((lstm_scan_idx + 1) * stride_4) + stride_5) + (((((int)blockIdx.x) * 24) + ((int)threadIdx.x)) * stride_6))] + s_h2h[1]))));
    }
    if (((int)threadIdx.y) == 0) {
      next_h[0] = ((1.000000e+00f / (1.000000e+00f + __expf((0.000000e+00f - (Xi2h[(((stride_5 * 3) + ((lstm_scan_idx + 1) * stride_4)) + (((((int)blockIdx.x) * 24) + ((int)threadIdx.x)) * stride_6))] + s_h2h[3]))))) * tanhf(next_c[0]));
    }
    if (((int)threadIdx.y) == 0) {
      lstm_scan[(((lstm_scan_idx + 1) * stride) + (((((int)blockIdx.x) * 24) + ((int)threadIdx.x)) * stride_1))] = next_h[0];
    }
    if (((int)threadIdx.y) == 0) {
      lstm_scan_1[(((lstm_scan_idx + 1) * stride_2) + (((((int)blockIdx.x) * 24) + ((int)threadIdx.x)) * stride_3))] = next_c[0];
    }
  }
}
                                                                                                                                                                         

