#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

using namespace nvcuda;

__global__ void matrixMul_Ampere(half *A, half *B, float *C, int M, int N, int K) {
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

  int blockRow = blockIdx.y * 16;
  int blockCol = blockIdx.x * 16;

    if (blockRow < 512 && blockCol < 512) {

      wmma::fill_fragment(c_frag, 0.0f);

      for (int k = 0; k < 512; k += 16) {

        wmma::load_matrix_sync(a_frag,
                               A + blockRow * 512 + k, 512);
        wmma::load_matrix_sync(b_frag,
                               B + k * 512 + blockCol, 512);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      }

      wmma::store_matrix_sync(C + blockRow * 512 + blockCol,
                              c_frag, 512, wmma::mem_row_major);
    }
}


__global__ void convertFloatToHalf(float* in, half* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = __float2half(in[idx]);
    }
}

int main() {
    int M = 512; // A 的行数
    int K = 512; // A 的列数 / B 的行数
    int N = 512; // B 的列数
    
    size_t size_A_float = M * K * sizeof(float);
    size_t size_B_float = K * N * sizeof(float);
    size_t size_C_float = M * N * sizeof(float);
    size_t size_A_half = M * K * sizeof(half);
    size_t size_B_half = K * N * sizeof(half);

    // 分配主机内存
    float *h_A = (float*)malloc(size_A_float);
    float *h_B = (float*)malloc(size_B_float);
    float *h_C = (float*)malloc(size_C_float);

    // 初始化矩阵 A 和 B
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = 1.0f; // 每个元素赋值为 1
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = 1.0f; // 每个元素赋值为 1
    }

    // 分配设备内存
    float *d_A_float, *d_B_float, *d_C;
    half *d_A_half, *d_B_half;
    cudaMalloc(&d_A_float, size_A_float);
    cudaMalloc(&d_B_float, size_B_float);
    cudaMalloc(&d_C, size_C_float);
    cudaMalloc(&d_A_half, size_A_half);
    cudaMalloc(&d_B_half, size_B_half);

    // 将数据从主机复制到设备
    cudaMemcpy(d_A_float, h_A, size_A_float, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_float, h_B, size_B_float, cudaMemcpyHostToDevice);

    // 转换浮点数为 half 类型
    int threadsPerBlock = 512;
    int blocksPerGrid_A = (M * K + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGrid_B = (K * N + threadsPerBlock - 1) / threadsPerBlock;
    convertFloatToHalf<<<blocksPerGrid_A, threadsPerBlock>>>(d_A_float, d_A_half, M * K);
    convertFloatToHalf<<<blocksPerGrid_B, threadsPerBlock>>>(d_B_float, d_B_half, K * N);

    // 定义线程块和网格大小
    dim3 threadsPerBlock2D(32);
    dim3 numBlocks((N + 16 - 1) / 16, (M + 16 - 1) / 16);
    for (int i = 0; i < 10; ++i) {
        matrixMul_Ampere<<<numBlocks, threadsPerBlock2D>>>(d_A_half, d_B_half, d_C, M, N, K);
    }
    // 定义 CUDA 事件以计算时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 启动内核
    cudaEventRecord(start);
    for (int i = 0; i < 1000; ++i) {
        matrixMul_Ampere<<<numBlocks, threadsPerBlock2D>>>(d_A_half, d_B_half, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // 计算执行时间
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    milliseconds = milliseconds / 1000.0f;
    float seconds = milliseconds / 1000.0f;
    long long total_flops = 2LL * M * N * K;
    float gflops = (total_flops / seconds) / 1e9f;

    printf("Matrix size: %d x %d and %d x %d\n", M, K, K, N);
    printf("Execution time: %f milliseconds\n", milliseconds);
    printf("Performance: %f GFLOPS\n", gflops);

    // 将结果从设备复制到主机
    cudaMemcpy(h_C, d_C, size_C_float, cudaMemcpyDeviceToHost);

    // 打印结果的前 10 个元素
    printf("First few elements of the result:\n");
    for (int i = 0; i < 10; ++i) {
        printf("%f ", h_C[i]);
    }
    printf("\n");

    // 清理内存
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A_float); cudaFree(d_B_float); cudaFree(d_C);
    cudaFree(d_A_half); cudaFree(d_B_half);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}

// nvcc -arch=sm_80 benchmark/source/gemm_tensorcore.cu -o output
// ./output

// Matrix size: 512 x 512 and 512 x 512
// Execution time: 0.020575 milliseconds
// Performance: 13046.533203 GFLOPS
