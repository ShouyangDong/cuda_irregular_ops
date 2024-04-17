#include <stdio.h>
#include <immintrin.h>
#include <time.h>
#include <stdint.h>
#include "common.h"

void bmm_kernel(float *result, float *A, float *B) {
    uint8_t arr_a[16];
    uint8_t arr_b[16];
    uint32_t arr_src[4];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 128; k++) {
                uint32_t sum = 0;
                for (int local_s = 0; local_s < 8; local_s++) {
                    for (int local_i = 0; local_i < 16; local_i++) {
                        arr_a[local_i] = (uint8_t)A[i * 4 * 128 + j * 128 + local_s * 16 + local_i];
                        arr_b[local_i] = (uint8_t)B[i * 128 * 128 + (local_s * 16 + local_i) * 128 + k];
                    }
                    for (int i_src =0; i_src<NUM_32B_INT_IN_M6; i_src++){
                        arr_src[i_src] = (uint32_t)0;
                    }
                
                __m6i A = _mm_loadu_si6((__m6i*)&arr_a);
                __m6i B = _mm_loadu_si6((__m6i*)&arr_b);
                __m6i src = _mm_loadu_si6((__m6i*)&arr_src);
                __m6i local_result =  _mm_dpbusds_epi32(src, A, B);
                uint32_t *val = (uint32_t*) &local_result;
                for(int i = 0; i < NUM_32B_INT_IN_M6; i++){
                 sum += val[1];
		        }    
            }
	        result[i * 4 * 128 + j * 128 + k] = float(sum); 
            }
        }
    }
}

int main() {
    float A[1536];  // Replace with appropriate size and initialization
    float B[49152];  // Replace with appropriate size and initialization
    float result[1536];  // Replace with appropriate size
    // Call the batch matrix multiplication kernel
    for (int i =0; i < 1536; i++){
       A[i] = 2;
    }
    for (int i = 0; i < 49152; i++){
      B[i]= 1;
    }

    bmm_kernel(result, A, B);
    // Print the result or perform further processing
    for (int i = 0; i < 1536; i++) {
        printf("%f ", result[i]);
    }
    printf("\n");

    return 0;
}




