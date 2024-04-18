#include <stdio.h>
#include <immintrin.h>
#include <time.h>
#include <stdint.h>
#include "common.h"

#define MAXRAND 10

__m128i A,B,src,result;

void foo(int a, int b, int c){

	uint8_t arr_a[NUM_8B_INT_IN_M128];
	uint8_t arr_b[NUM_8B_INT_IN_M128];
	uint32_t arr_src[4];

    for (int i=0; i<NUM_8B_INT_IN_M128; i++){
        arr_a[i] = (uint8_t)a;
        arr_b[i] = (uint8_t)b;
	}

    for (int i=0; i<4; i++){
        arr_src[i] = c;
	}
	A = _mm_loadu_si128((__m128i*)&arr_a);
	B = _mm_loadu_si128((__m128i*)&arr_b);
	src = _mm_loadu_si128((__m128i*)&arr_src);

	result =  _mm_dpbusds_epi32(src,A,B);

}

int main(int argc, char **argv){

	srand((unsigned)time(NULL));
	int a =  2;
	int b =  4;
	int c =  0;

    foo(a,b,c);

	printf("A = ");
	print128_num_8(A);

	printf("B = ");
	print128_num_8(B);

	printf("src = ");
	print128_num_32(src);

	printf("result = ");
	print128_num_32(result);

    return 0;
}
