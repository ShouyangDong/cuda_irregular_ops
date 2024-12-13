#include <chrono>
#include <random>
#include <iostream>
#include <stdio.h>
#include <immintrin.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
#include <omp.h>

#define M 32
#define N 128
#define kThreads 4

void gemm_kernel(float *A, float *B, float *result) {
  uint8_t arr_a[64];
  uint8_t arr_b[64];
  uint32_t arr_d[16];
    #pragma omp parallel for
  for (int j = 0; j < 32; j++) {
    for (int k = 0; k < 128; k++) {
      uint32_t sum = 0;
      // 使用VNNI指令进行乘加操作
      __m512i acc = _mm512_setzero_si512(); // 初始化累加器为0
      // 遍历128个元素，每次处理64个，以适应AVX512
      for (int local_s = 0; local_s < 2; local_s++) {
        // 将浮点数组A和B量化到int8类型
        for (int local_i = 0; local_i < 64; local_i++) {
          arr_a[local_i] =
              static_cast<uint8_t>(A[j * 128 + local_s * 64 + local_i]);
          arr_b[local_i] =
              static_cast<uint8_t>(B[(local_s * 64 + local_i) * 128 + k]);
        }

        // 加载量化后的数据到512位SIMD寄存器中
        __m512i _a = _mm512_loadu_si512(reinterpret_cast<const void *>(arr_a));
        __m512i _b = _mm512_loadu_si512(reinterpret_cast<const void *>(arr_b));

        // 使用_mm512_dpbusd_epi32进行乘加操作 (AVX512 VNNI)
        acc = _mm512_dpbusd_epi32(acc, _a, _b); // 执行乘加操作：acc += a * b
      }

      // 将累加结果存储到arr_d中
      _mm512_storeu_si512(reinterpret_cast<void *>(arr_d), acc);

      // 将arr_d中的值累加得到最终的结果
      for (int i = 0; i < 16; ++i) {
        sum += arr_d[i];
      }

      // 反量化并存储到输出矩阵result中
      result[j * 128 + k] = static_cast<float>(sum);
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


// g++ -march=icelake-server -fopenmp -O3 -o test_program benchmark/source/unroll_gemm_dlboost.cpp
// ./test_program

// Matrix size: 32x128 and 128x128 
// Execution time: 31.0935 milliseconds
// Performance: 3.37233e-05 GFLOPS
