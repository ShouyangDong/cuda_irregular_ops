#include <chrono>
#include <random>
#include <iostream>
#include <stdio.h>
#include <immintrin.h>
#include <stdint.h>
#include <math.h>
#include <float.h>

#define M 128
#define N 128
#define block_size 16

void matmul(float *A, float *B, float *C) {
    #pragma omp parallel for
    for (int i = 0; i < N; i += block_size) {
        for (int j = 0; j < N; j += block_size) {
            for (int k = 0; k < N; k += block_size) {
                for (int ii = i; ii < i + block_size; ++ii) {
                    for (int kk = k; kk < k + block_size; ++kk) {
                        __m512 a = _mm512_load_ps(&A[ii*N + kk]);
                        for (int jj = j; jj < j + block_size; jj += 16) {
                            __m512 b = _mm512_load_ps(&B[kk*N + jj]);
                            __m512 c = _mm512_load_ps(&C[ii*N + jj]);
                            c = _mm512_fmadd_ps(a, b, c);
                            _mm512_store_ps(&C[ii*N + jj], c);
                        }
                    }
                }
            }
        }
    }
}


int main() {
    float* A = new float[N*N];
    float* B = new float[N*N];
    float* C = new float[N*N];

    // Fill A and B with random numbers
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(1.0, 2.0);
    for(int i = 0; i < N*N; ++i) {
        A[i] = dis(gen);
        B[i] = dis(gen);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i =0 ;i < 1000;i++) {
        matmul(A, B, C);
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end-start;
    float total_flops = 2 * N * N * N;
    float gflops = (total_flops / diff.count()) / 1e9f;
    std::cout << "Matrix size: " << N << "x" << N << " and " << N << "x" << N << " \n";
    std::cout << "Execution time: " << diff.count() * 1000 / 1000.0f << " milliseconds\n";
    std::cout << "Performance: " << gflops << " GFLOPS\n";

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
// g++ -fopenmp -o matmul benchmark/optimization/dlboost/gemm_omp.cpp
// ./matmul 
// Matrix size: 128x128 and 128x128 
// Execution time: 0.20937 milliseconds
// Performance: 0.020033 GFLOPS
