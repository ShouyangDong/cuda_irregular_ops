CUDA
```
extern "C" __global__ void matmul(half *A, half *B, float *D)
{
	int ix = (blockIdx.x * blockDim.x + threadIdx.x)/32;
	int iy = (blockIdx.y * blockDim.y + threadIdx.y);
	
	wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> ab_frag;
	
	wmma::fill_fragment(ab_frag, 0.0f);

	int a_row = ix * 16;
	int b_row = iy * 16;
	for (int k=0; k<512; k+=16) {
		int a_col = k;
        int b_col = k;

		if (a_row < 512 && a_col < 512 && b_row < 512 && b_col < 512) {
			// Load the inputs
			wmma::load_matrix_sync(a_frag, A + a_col + a_row * 512, 512);
			wmma::load_matrix_sync(b_frag, B + b_col + b_col * 512, 512);

			// Perform the matrix multiplication
			wmma::mma_sync(ab_frag, a_frag, b_frag, ab_frag);
		}
	}

	if (a_row < 512 && b_row < 512) {
		// Store the output
		wmma::store_matrix_sync(D + b_row + a_row * N_TOTAL, ab_frag, N_TOTAL, wmma::mem_row_major);
	}
}
```
MLU
```
extern "C" __mlu_global__ void matmul(half* data, half* filter, float* output) {
  __nram__ half date_block[768];
  __wram__ half filter_block[65536];
  for (int i = 0; i < 64; ++i) {
    for (int j = 0; j < 8; ++j) {
      __memcpy(date_block, data + i * 4096 + j * 512, 1024, GDRAM2NRAM);
      for (int k = 0; k < 4; ++k) {
        __memcpy(filter_block, filter + k * 128, 256, GDRAM2WRAM, 256, 1024, 511);
        __bang_mlp(date_block + 256, date_block, filter_block, 512, 128, 0);
        __memcpy(output i * 4096 + j * 512 + k * 128, date_block + 256, 512, NRAM2GDRAM);
      }
    }
  }
}
```
HIP
```
extern "C" __global__ void matmul(half *A, half *B, float *C) {
    using float16x4 = __attribute__((__vector_size__(4 * sizeof(float16_t)))) float16_t;
    using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

    const int c_row_base = blockIdx.y * 16;
    const int c_col_base = blockIdx.x * 16;

    floatx4 d = {0.0f};
    for(int k_step = 0; k_step < 512; k_step += 16) {
        float16x4 a, b;
        for(int i = 0; i < 4; ++i) {

            int a_row = c_row_base + threadIdx.x;
            int a_col = k_step + threadIdx.y * 4 + i;
            a[i] = A[a_row * 512 + a_col];

            int b_row = k_step + threadIdx.y * 4 + i;
            int b_col = c_col_base + threadIdx.x;
            b[i] = B[b_row * 512 + b_col];
        }

        d = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, d, 0, 0, 0);
    }

    for(int i = 0; i < 4; ++i) {
        int c_row = c_row_base + threadIdx.x;
        int c_col = c_col_base + threadIdx.y * 4 + i;
        if(c_row < 512 && c_col < 512) {
            C[c_row * 512 + c_col] = d[i];
        }
    }
}
```
DL Boost
```
extern "C"  void matmul(const int8_t* A, const int8_t* B, int32_t* C) {
    #pragma omp parallel for
    for (int i = 0; i < 512; ++i) {
        for (int j = 0; j < 512; j += 16) {
            __m512i acc = _mm512_setzero_si512();
            for (int p = 0; p < 512; ++p) {
                __m512i a_vec = _mm512_set1_epi8(A[i * 512 + p]); 

                __m128i b_vec_128 = _mm_loadu_si128((__m128i*)&B[p * 512 + j]);
                __m512i b_vec_512 = _mm512_cvtepi8_epi32(b_vec_128);

                acc = _mm512_dpbusds_epi32(acc, a_vec, b_vec_512);
            }

            _mm512_storeu_si512((__m512i*)&C[i * 512 + j], acc);
        }
    }
}
``·