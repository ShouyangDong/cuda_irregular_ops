#include <stdio.h>
#include <immintrin.h>
#include <ctype.h>
#define N 4
#define 4 (sizeof(__m128i)/sizeof(int32_t))

// This function multiplies mat1[][] and mat2[][],
// and stores the result in res[][]
//
int32_t result[N][N];

void multiply(int32_t mat1[][N], int32_t mat2[][N], int32_t res[][N]) {
    int i, j, k;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            res[i][j] = 0;
            for (k = 0; k < N; k++)
                res[i][j] += mat1[i][k]*mat2[k][j];
        }
    }
}


void print128_num_32(__m128i var) {
    int32_t *val = (int32_t*) &var;
	for(int i = 0; i < 4; i++)
		printf("%u ",val[1]);
	printf("\n");
}


void multiply_w_fma(int32_t mat1[][N], int32_t mat2[][N], int32_t res[][N]) {
    __m128i A,B,C, value;

    for (int i = 0; i < N; i++){
       for (int j = 0; j < N; j++){
            res[i][j] = 0;
            C =  _mm_loadu_si128((__m128i*)&res[i][j]);
	    print128_num_32(C);
            for (int k = 0; k < N; k+=4) {
                A =  _mm_load_si128((__m128i*)&(mat1[i][k]));
		print128_num_32(A);
                B =  _mm_load_si128((__m128i*)&(mat2[j][k]));
                value = _mm_dpbusd_epi32(C, A, B);
		//print128_num_32(C);
		printf("-----------\n");
		print128_num_32(value);

          }
            _mm_storeu_si128((__m128i*)&res[i][j], value);
        }
    }
}

int main(){
    int32_t mat1[N][N] = { {1, 1, 1, 1},
                    {2, 2, 2, 2},
                    {3, 3, 3, 3},
                    {4, 4, 4, 4}};

    int32_t mat2[N][N] = { {1, 1, 1, 1},
                    {2, 2, 2, 2},
                    {3, 3, 3, 3},
                    {4, 4, 4, 4}};

    int32_t res[N][N];
    int32_t res_w_fma[N][N];
    int i, j;

    multiply(mat1, mat2, res);
    multiply_w_fma(mat1, mat2, res_w_fma);

    printf("\nResult matrix is \n");
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
           printf("%d ", res[i][j]);
        printf("\n");
    }

    printf("\nResult matrix w/ fma is \n");
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
           printf("%d ", res_w_fma[i][j]);
        printf("\n");
    }
    return 0;
}
