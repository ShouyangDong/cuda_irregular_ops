#include <tl_templates/gemm.h>
#include <tl_templates/copy.h>
#include <tl_templates/reduce.h>
#include <tl_templates/threadblock_swizzle.h>

extern "C" __global__ void __launch_bounds__(128) main_kernel(float* __restrict__ A, float* __restrict__ B) {
  extern __shared__ uchar buf_dyn_shmem[];
  float A_local[64];
  float A_powsum[1];
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    *(float4*)(((float*)buf_dyn_shmem) + (((i * 512) + (((int)threadIdx.x) * 4)) + 128)) = *(float4*)(A + (((((int)blockIdx.x) * 8192) + (i * 512)) + (((int)threadIdx.x) * 4)));
  }
  __syncthreads();
  #pragma unroll
  for (int i_1 = 0; i_1 < 16; ++i_1) {
    float4 __1;
      float4 v_ = *(float4*)(((float*)buf_dyn_shmem) + (((i_1 * 512) + (((int)threadIdx.x) * 4)) + 128));
      __1.x = (v_.x*v_.x);
      __1.y = (v_.y*v_.y);
      __1.z = (v_.z*v_.z);
      __1.w = (v_.w*v_.w);
    *(float4*)(A_local + (i_1 * 4)) = __1;
  }
  A_powsum[0] = 0.000000e+00f;
  #pragma unroll
  for (int rv = 0; rv < 64; ++rv) {
    A_powsum[0] = (A_powsum[0] + A_local[(((rv & 15) * 4) + (rv >> 4))]);
  }
  __syncthreads();
  A_powsum[0] = tl::AllReduce<tl::SumOp, 128, 1>::run(A_powsum[0], (&(((float*)buf_dyn_shmem)[0])));
  A_powsum[0] = ((1.000000e+00f / sqrtf((A_powsum[0] * 1.220703e-04f))) + 1.000000e-12f);
  __syncthreads();
  #pragma unroll
  for (int i_2 = 0; i_2 < 64; ++i_2) {
    ((float*)buf_dyn_shmem)[(((i_2 * 128) + ((int)threadIdx.x)) + 128)] = (((float*)buf_dyn_shmem)[(((i_2 * 128) + ((int)threadIdx.x)) + 128)] * A_powsum[0]);
  }
  __syncthreads();
  #pragma unroll
  for (int i_3 = 0; i_3 < 16; ++i_3) {
    *(float4*)(B + (((((int)blockIdx.x) * 8192) + (i_3 * 512)) + (((int)threadIdx.x) * 4))) = *(float4*)(((float*)buf_dyn_shmem) + (((i_3 * 512) + (((int)threadIdx.x) * 4)) + 128));
  }
}
