extern "C" void bmm_kernel(float *A, float *B, float *result) {
  uint8_t arr_a[16];
  uint8_t arr_b[16];
  uint32_t arr_src[4];
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 32; j++) {
      for (int k = 0; k < 6; k++) {
        uint32_t sum = 0;
        for (int local_s = 0; local_s < 8; local_s++) {
          for (int local_i = 0; local_i < 16; local_i++) {
            arr_a[local_i] =
                (uint8_t)A[i * 32 * 128 + j * 128 + local_s * 16 + local_i];
            arr_b[local_i] =
                (uint8_t)B[i * 128 * 6 + (local_s * 16 + local_i) * 6 + k];
          }
          for (int i_src = 0; i_src < 4; i_src++) {
            arr_src[i_src] = (uint32_t)0;
          }

          __m128i A = _mm_loadu_si128((__m128i *)&arr_a);
          __m128i B = _mm_loadu_si128((__m128i *)&arr_b);
          __m128i src = _mm_loadu_si128((__m128i *)&arr_src);
          __m128i local_result = _mm_dpbusds_epi32(src, A, B);
          uint32_t *val = (uint32_t *)&local_result;
          for (int i = 0; i < 4; i++) {
            sum += val[i];
          }
        }
        result[i * 32 * 6 + j * 6 + k] = float(sum);
      }
    }
  }
}
