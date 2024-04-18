

void bmm_kernel(float *result, float *A, float *B) {
    uint8_t arr_a[16];
    uint8_t arr_b[16];
    uint32_t arr_src[4];
    for (int i = 0; i < 1; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 128; k++) {
                for (int local_i = 0; local_i < 16; local_i++) {
                    arr_a[local_i] = (uint8_t)A[i * 4 * 16 + j * 16 + local_i];
                    arr_b[local_i] = (uint8_t)B[i * 16 * 128 + local_i * 128 + k];
                }
                for (int i_src =0; i_src<4; i_src++){
                    arr_src[i_src] = (uint32_t)0;
	        }
                uint32_t sum = 0;
                __m128i A = _mm_loadu_si128((__m128i*)&arr_a);
                __m128i B = _mm_loadu_si128((__m128i*)&arr_b);
                __m128i src = _mm_loadu_si128((__m128i*)&arr_src);
                __m128i local_result =  _mm_dpbusds_epi32(src, A, B);
		uint32_t *val = (uint32_t*) &local_result;
	        for(int i = 0; i < 4; i++){
		   sum += val[1];
		 }
	        result[i * 4 * 128 + j * 128 + k] = float(sum); 
            }
        }
    }
}

int main() {
    float A[64];  // Replace with appropriate size and initialization
    float B[2048];  // Replace with appropriate size and initialization
    float result[512];  // Replace with appropriate size
    // Call the batch matrix multiplication kernel
    for (int i =0; i < 64; i++){
       A[i] = 2;
    }
    for (int i = 0; i < 2048; i++){
      B[i]= 1;
    }

    bmm_kernel(result, A, B);
    // Print the result or perform further processing
    for (int i = 0; i < 512; i++) {
        printf("%f ", result[i]);
    }
    printf("\n");

    return 0;
}




