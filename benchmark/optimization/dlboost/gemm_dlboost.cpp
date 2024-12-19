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


#include <immintrin.h>
#include <cstdint>
#include <omp.h>

void gemm_kernel(float *A, float *B, float *result) {
  #pragma omp parallel for
  for (int j = 0; j < 128; j++) {
    for (int k = 0; k < 128; k++) {
      // 初始化累加器
      __m512i acc = _mm512_setzero_si512();

      // 遍历 128 个元素，每次处理 64 个
      for (int local_s = 0; local_s < 2; local_s++) {
        // 加载 64 个浮点数
        __m512 _a_f32 = _mm512_loadu_ps(&A[j * 128 + local_s * 64]);
        __m512 _b_f32 = _mm512_loadu_ps(&B[(local_s * 64) * 128 + k]);

        // 1. 转换 float -> int32
        __m512i _a_int32 = _mm512_cvtps_epi32(_a_f32);
        __m512i _b_int32 = _mm512_cvtps_epi32(_b_f32);

        // 2. 转换 int32 -> int16
        __m256i _a_int16 = _mm512_cvtusepi32_epi16(_a_int32);
        __m256i _b_int16 = _mm512_cvtusepi32_epi16(_b_int32);

        // 3. 转换 int16 -> int8
        __m128i _a_int8 = _mm256_cvtusepi16_epi8(_a_int16);
        __m128i _b_int8 = _mm256_cvtusepi16_epi8(_b_int16);

        // 扩展 int8 -> int32（适配 VNNI 指令）
        __m512i _a_packed = _mm512_cvtepu8_epi32(_a_int8);
        __m512i _b_packed = _mm512_cvtepu8_epi32(_b_int8);

        // 使用 VNNI 指令进行乘加：acc += _a_packed * _b_packed
        acc = _mm512_dpbusd_epi32(acc, _a_packed, _b_packed);
      }

      // 使用水平加法快速完成归约
      int result_sum = _mm512_reduce_add_epi32(acc);
      result[j * 128 + k] = static_cast<float>(result_sum);
    }
  }
}





void random_matrix(float *A, int rows, int cols) {
  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(0.0, 1.0);

  for (int i = 0; i < rows * cols; ++i) {
    A[i] = 1;
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
  for (int i =0; i < 10;i++) {
      std::cout << "number: " << C[i];
  }
  std::cout << "Matrix size: " << M << "x" << N << " and " << N << "x" << N << " \n";
  std::cout << "Execution time: " << diff.count() * 1000 / 1000.0f << " milliseconds\n";
  std::cout << "Performance: " << gflops << " GFLOPS\n";
  delete[] A;
  delete[] B;
  delete[] C;

  return 0;
}


// g++ -march=icelake-server -O3 -fopenmp -o test_program benchmark/optimization/dlboost/gemm_dlboost.cpp
// ./test_program
// Matrix size: 128x128 and 128x128 
// Execution time: 0.0214131 milliseconds
// Performance: 0.195876 GFLOPS
