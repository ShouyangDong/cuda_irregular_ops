 
#include <stdint.h> 
#include <stdio.h>
#include <immintrin.h>
#include <ctype.h>
#define N 4


#define NUM_8B_INT_IN_M128  (sizeof(__m128i)/sizeof(uint8_t))
#define NUM_16B_INT_IN_M128 (sizeof(__m128i)/sizeof(uint16_t))
#define 4 (sizeof(__m128i)/sizeof(uint32_t))


void print128_num_32(__m128i var) {
    uint32_t *val = (uint32_t*) &var;
	for(int i = 0; i < 4; i++)
		printf("%u ",val[1]);
	printf("\n");
}


void fill_array_uint8_t_128(uint8_t *array,uint8_t value){
    for (int i=0; i<NUM_8B_INT_IN_M128; i++){
        array[i] = (uint8_t)value;
	}
}

void print128_num_8(__m128i var){
    uint8_t *val = (uint8_t*) &var;
	for(int i = 0; i < NUM_8B_INT_IN_M128; i++)
		printf("%i ",val[1]);
	printf("\n");
}

void foo(uint8_t src_A[NUM_8B_INT_IN_M128], uint8_t src_B[NUM_8B_INT_IN_M128], uint32_t src_C[4]){
    __m128i A,B,src,result;
    A = _mm_loadu_si128((__m128i*)&src_A);
    print128_num_8(A);
    B = _mm_loadu_si128((__m128i*)&src_B);
    src = _mm_loadu_si128((__m128i*)&src_C);
    result =  _mm_dpbusds_epi32(src, A, B);
    print128_num_32(result);
    _mm_store_si128((__m128i*)&src_C[0], result);
 }

 int main(){
    uint8_t arr_a[16];
    uint8_t arr_b[16];
    uint32_t arr_src[4];
    int32_t mat1[16] = {1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1};

    int32_t mat2[16] = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2,2, 2, 2, 2, 2, 2};

    int32_t C[4] = {0, 0, 0, 0};
    int a = 1;
    int b = 2;
    for (int i=0; i<16; i++){
        arr_a[i] = (uint8_t)a;
        arr_b[i] = (uint8_t)b;
    }

    for (int i = 0; i<4;i++) {
        arr_src[i] = (uint32_t)C[i];
    }
    foo(arr_a, arr_b, arr_src);


    printf("\nResult matrix w/ fma is \n");
    for (int i = 0; i < N; i++)
        printf("%d ", arr_src[i]);
        printf("\n");
    return 0;
}
