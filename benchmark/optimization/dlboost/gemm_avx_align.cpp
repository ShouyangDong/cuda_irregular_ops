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

void gemm_kernel(float *A, float *B, float *C) {
    const int block_size = 32; // 适当的块大小，依赖于具体的CPU缓存大小
    // 使用对齐分配来确保内存访问的效率
    float* B_block = (float*)_mm_malloc(block_size * N * sizeof(float), 64);

    #pragma omp parallel for
    for (int jj = 0; jj < N; jj += block_size) {
        for (int kk = 0; kk < N; kk += block_size) {
            // 将B的块复制到连续的内存中
            for (int i = 0; i < block_size; ++i) {
                for (int j = 0; j < N; ++j) {
                    B_block[i * N + j] = B[(kk + i) * N + jj + j];
                }
            }

            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < block_size; ++j) {
                    __m512 sum = _mm512_setzero_ps();
                    for (int k = 0; k < block_size; k += 16) {
                        // 软件预取
                        _mm_prefetch((const char*)&A[(i * N) + kk + k + 16], _MM_HINT_T0);
                        _mm_prefetch((const char*)&B_block[(k + 16) * N + j], _MM_HINT_T0);
                        
                        __m512 a = _mm512_load_ps(&A[i * N + kk + k]);
                        __m512 b = _mm512_load_ps(&B_block[k * N + j]);
                        sum = _mm512_fmadd_ps(a, b, sum); // FMA
                    }
                    C[i * N + jj + j] += _mm512_reduce_add_ps(sum);
                }
            }
        }
    }

    _mm_free(B_block);
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


// g++ -march=icelake-server -o test_program benchmark/optimization/dlboost/gemm_dlboost.cpp
// /test_program
// Matrix size: 128x128 and 128x128 
// Execution time: 9.28553 milliseconds
// Performance: 0.000451703 GFLOPS

