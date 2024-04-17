extern "C" void gemv_kernel(float *y, float *A, float *x) {
    uint8_t arr_a[16];
    uint8_t arr_b[16];
    uint32_t arr_src[4];
    for (int i = 0; i < 32; i++) {
        uint32_t sum = 0;
        for (int local_s = 0; local_s < 32; local_s++) {
            for (int local_i = 0; local_i < 16; local_i++) {
                arr_a[local_i] = (uint8_t)A[i * 64 + (local_s * 16 + local_i)];
                arr_b[local_i] = (uint8_t)B[local_s * 16 + local_i];
            }
            for (int i_src =0; i_src<NUM_32B_INT_IN_M6; i_src++){
                arr_src[i_src] = (uint32_t)0;
            }
            
            __m128i A = _mm_loadu_si128((__m128i*)&arr_a);
            __m128i B = _mm_loadu_si128((__m128i*)&arr_b);
            __m128i src = _mm_loadu_si128((__m128i*)&arr_src);
            __m128i local_result =  _mm_dpbusds_epi32(src, A, B);
            uint32_t *val = (uint32_t*) &local_result;
            for(int i = 0; i < NUM_32B_INT_IN_M128; i++){
                sum += val[1];
            }
        }
        y[i] = float(sum); 
    }
}