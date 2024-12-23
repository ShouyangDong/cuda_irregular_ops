#include <chrono>
#include <random>
#include <iostream>
#include <immintrin.h>
#include <omp.h>
#include <cstdint>

const int M = 128;
const int N = 128;
const int K = 128;

void gemm_kernel(const int8_t* A, const int8_t* B, float* C) {
    #pragma omp parallel for
    for (int i = 0; i < M; i += 4) {
        for (int j = 0; j < N; j += 64) {
            // 初始化 c 矩阵寄存器
            __m512i c[4] = {
                _mm512_setzero_si512(),
                _mm512_setzero_si512(),
                _mm512_setzero_si512(),
                _mm512_setzero_si512()
            };

            // 遍历 k 维
            for (int k = 0; k < K; ++k) {
                // 每次加载一个标量扩展为 512 位
                __m512i a0_vec = _mm512_set1_epi8(A[(i + 0) * K + k]);
                __m512i a1_vec = _mm512_set1_epi8(A[(i + 1) * K + k]);
                __m512i a2_vec = _mm512_set1_epi8(A[(i + 2) * K + k]);
                __m512i a3_vec = _mm512_set1_epi8(A[(i + 3) * K + k]);

                // 加载 B 的一行 (16 元素，按 int8_t 加载)
                __m512i b = _mm512_loadu_si512((__m512i*)&B[k * N + j]);

                // 累积点积计算
                c[0] = _mm512_dpbusds_epi32(c[0], a0_vec, b);
                c[1] = _mm512_dpbusds_epi32(c[1], a1_vec, b);
                c[2] = _mm512_dpbusds_epi32(c[2], a2_vec, b);
                c[3] = _mm512_dpbusds_epi32(c[3], a3_vec, b);
            }

            // 转换为 float 并存储
            for (int x = 0; x < 4; ++x) {
                __m512 result = _mm512_cvtepi32_ps(c[x]);
                // 修正结果，除以4
                result = _mm512_mul_ps(result, _mm512_set1_ps(0.25f));
                _mm512_storeu_ps(&C[(i + x) * N + j], result);
            }
        }
    }
}





int main() {
    int8_t* A = (int8_t*)_mm_malloc(M * K * sizeof(int8_t), 64);
    int8_t* B = (int8_t*)_mm_malloc(N * K * sizeof(int8_t), 64);
    float* C = (float*)_mm_malloc(M * N * sizeof(float), 64);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(-128, 127);
    for (int i = 0; i < M * K; ++i) {
        A[i] = 1;
    }
    for (int i = 0; i < N * K; ++i) {
        B[i] = 1;
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++) {
        gemm_kernel(A, B, C);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end-start;
    float total_flops = 2 * N * N * N;
    float gflops = (total_flops / diff.count()) / 1e9f;
    for (int i = 0; i < 10;i++) {
        std::cout << "number: " << C[i];
    }
    std::cout << "Matrix size: " << N << "x" << N << " and " << N << "x" << N << " \n";
    std::cout << "Execution time: " << diff.count() * 1000 / 1000.0f << " milliseconds\n";
    std::cout << "Performance: " << gflops << " GFLOPS\n";

    _mm_free(A);
    _mm_free(B);
    _mm_free(C);

    return 0;
}
