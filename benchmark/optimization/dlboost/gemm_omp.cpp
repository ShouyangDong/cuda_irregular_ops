#include <iostream>
#include <chrono>
#include <random>
#include <omp.h>

void matmul(float* A, float* B, float* C, int N) {
    #pragma omp parallel for
    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < N; ++j) {
            float sum = 0;
            for(int k = 0; k < N; ++k) {
                sum += A[i*N + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
}

int main() {
    const int N = 128;
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
        matmul(A, B, C, N);
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
