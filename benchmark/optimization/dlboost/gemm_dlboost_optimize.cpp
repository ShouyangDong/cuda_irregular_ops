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

void gemm_kernel(float *A, float *B, float *result) {
    const int block_size = 32;  // 块大小可以根据实际情况调整以获得最佳性能
    #pragma omp parallel for collapse(2)
    for (int jj = 0; jj < N; jj += block_size) {
        for (int kk = 0; kk < N; kk += block_size) {
            for (int i = 0; i < N; ++i) {
                for (int j = jj; j < std::min(jj + block_size, N); ++j) {
                    __m512 sum = _mm512_setzero_ps();
                    for (int k = kk; k < std::min(kk + block_size, N); k += 16) {
                        __m512 a = _mm512_loadu_ps(A + i * N + k);
                        __m512 b = _mm512_loadu_ps(B + k * N + j);
                        sum = _mm512_fmadd_ps(a, b, sum);  // acc += a * b
                    }
                    result[i * N + j] += _mm512_reduce_add_ps(sum);
                }
            }
        }
    }
}


void random_matrix(float *A, int rows, int cols) {
  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(0.0, 1.0);

  for (int i = 0; i < rows * cols; ++i) {
    A[i] = distribution(generator);
  }
}

int main() {
  float *A = new float[M * N];
  float *B = new float[N * N];
  float *C = new float[M * N];

  random_matrix(A, M, N);
  random_matrix(B, N, N);
  for(int i =0; i < 10;i++) {
    gemm_kernel(A, B, C);
  }

  auto start = std::chrono::high_resolution_clock::now();
  for(int i =0; i < 1000;i++) {
    gemm_kernel(A, B, C);
  }
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> diff = end - start;
float total_flops = 2 * M * N * N;
float gflops = (total_flops / diff.count()) / 1e9f;
  std::cout << "Matrix size: " << M << "x" << N << " and " << N << "x" << N << " \n";
  std::cout << "Execution time: " << diff.count() * 1000 / 1000.0f << " milliseconds\n";
  std::cout << "Performance: " << gflops << " GFLOPS\n";
  delete[] A;
  delete[] B;
  delete[] C;

  return 0;
}


// g++ -march=icelake-server -O3 -o test_program benchmark/optimization/dlboost/gemm_dlboost_optimize.cpp
// ./test_program
// Matrix size: 128x128 and 128x128 
// Execution time: 0.0727115 milliseconds
// Performance: 0.0576842 GFLOPS
