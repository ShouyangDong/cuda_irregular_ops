#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>

using namespace nvcuda;

__global__ void matrixMul_Ampere(half *A, half *B, float *C, int M, int N, int K) {
    // 每个线程块计算 C 中的一个 16x16 块
    // 定义 wmma fragment
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;


    // 确保线程块范围在矩阵尺寸内
    if (blockIdx.y * 16 < M || blockIdx.x * 16 < N) {
        // 初始化 C 的片段为 0
        wmma::fill_fragment(c_frag, 0.0f);

        // 在 K 方向上迭代，每次处理 16 个元素
        for (int k = 0; k < K / 16; k ++) {
            // 加载 A 和 B 中的片段
            wmma::load_matrix_sync(a_frag, A + blockIdx.y * 16 * K + k * 16, K);
            wmma::load_matrix_sync(b_frag, B + k * 16 * N + blockIdx.x * 16, N);

            // 计算片段并累加结果
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        // 将计算结果存储到矩阵 C 中
        wmma::store_matrix_sync(C + (blockIdx.y * 16 * N + blockIdx.x * 16), c_frag, N, wmma::mem_row_major);
    }
}

__global__ void convertFloatToHalf(float* in, half* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = __float2half(in[idx]);
    }
}

int main() {
    int N = 1024; // 定义矩阵大小
    size_t size_float = N * N * sizeof(float);
    size_t size_half = N * N * sizeof(half);

    // 分配主机内存
    float *h_A = (float*)malloc(size_float);
    float *h_B = (float*)malloc(size_float);
    float *h_C = (float*)malloc(size_float);

    // 初始化矩阵 A 和 B
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = 1.0f; // 每个元素赋值为 1
        h_B[i] = 1.0f; // 每个元素赋值为 1
    }

    // 分配设备内存
    float *d_A_float, *d_B_float, *d_C;
    half *d_A_half, *d_B_half;
    cudaMalloc(&d_A_float, size_float);
    cudaMalloc(&d_B_float, size_float);
    cudaMalloc(&d_C, size_float);
    cudaMalloc(&d_A_half, size_half);
    cudaMalloc(&d_B_half, size_half);

    // 将数据从主机复制到设备
    cudaMemcpy(d_A_float, h_A, size_float, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_float, h_B, size_float, cudaMemcpyHostToDevice);

    // 转换浮点数为 half 类型
    int threadsPerBlock = 256;
    int blocksPerGrid = (N * N + threadsPerBlock - 1) / threadsPerBlock;
    convertFloatToHalf<<<blocksPerGrid, threadsPerBlock>>>(d_A_float, d_A_half, N * N);
    convertFloatToHalf<<<blocksPerGrid, threadsPerBlock>>>(d_B_float, d_B_half, N * N);

    // 定义线程块和网格大小
    dim3 threadsPerBlock2D(16, 16);  // 每个块 32 x 8 个线程
    dim3 numBlocks((N + 16 - 1) / 16, (N + 16 - 1) / 16);  // 确保覆盖所有 16x16 的子块

    // 定义 CUDA 事件以计算时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 启动内核
    cudaEventRecord(start);
    matrixMul_Ampere<<<numBlocks, threadsPerBlock2D>>>(d_A_half, d_B_half, d_C, N, N, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // 计算执行时间
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float seconds = milliseconds / 1000.0f;
    long long total_flops = 2LL * N * N * N;
    float gflops = (total_flops / seconds) / 1e9f;

    printf("Matrix size: %d x %d\n", N, N);
    printf("Execution time: %f milliseconds\n", milliseconds);
    printf("Performance: %f GFLOPS\n", gflops);

    // 将结果从设备复制到主机
    cudaMemcpy(h_C, d_C, size_float, cudaMemcpyDeviceToHost);

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
