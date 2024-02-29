#include <tl_templates/gemm.h>
#include <tl_templates/copy.h>
#include <tl_templates/reduce.h>
#include <tl_templates/threadblock_swizzle.h>

extern "C" __global__ void __launch_bounds__(256) main_kernel(float* __restrict__ ins, float* __restrict__ outs) {
  extern __shared__ float4 shared[];
  float local[16];
  float local_t[16];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    shared[(((i * 272) + ((((int)threadIdx.x) >> 4) * 17)) + (((int)threadIdx.x) & 15))] = *(float4*)(ins + (((((((int)blockIdx.y) * 524288) + (i * 131072)) + ((((int)threadIdx.x) >> 4) * 8192)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 15) * 4)));
  }
  __syncthreads();
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    *(float4*)(local + (i_1 * 4)) = shared[((((((int)threadIdx.x) >> 4) * 68) + (i_1 * 17)) + (((int)threadIdx.x) & 15))];
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 16; ++i_2) {
    local_t[i_2] = local[(((i_2 & 3) * 4) + (i_2 >> 2))];
  }
  __syncthreads();
  #pragma unroll
  for (int i_3 = 0; i_3 < 4; ++i_3) {
    shared[((((((int)threadIdx.x) & 15) * 68) + (i_3 * 17)) + (((int)threadIdx.x) >> 4))] = *(float4*)(local_t + (i_3 * 4));
  }
  __syncthreads();
  #pragma unroll
  for (int i_4 = 0; i_4 < 4; ++i_4) {
    *(float4*)(outs + (((((((int)blockIdx.x) * 524288) + (i_4 * 131072)) + ((((int)threadIdx.x) >> 4) * 8192)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) & 15) * 4))) = shared[(((i_4 * 272) + ((((int)threadIdx.x) >> 4) * 17)) + (((int)threadIdx.x) & 15))];
  }
}
