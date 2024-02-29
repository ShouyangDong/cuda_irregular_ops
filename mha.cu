#include <tl_templates/gemm.h>
#include <tl_templates/copy.h>
#include <tl_templates/reduce.h>
#include <tl_templates/threadblock_swizzle.h>

extern "C" __global__ void __launch_bounds__(128) main_kernel(half_t* __restrict__ K, half_t* __restrict__ Output, half_t* __restrict__ Q, half_t* __restrict__ V) {
  extern __shared__ uchar buf_dyn_shmem[];
  float acc_o[128];
  float logsum[2];
  float scores_max[2];
  half_t Q_local[128];
  float acc_s[16];
  float scores_max_prev[2];
  float scores_scale[2];
  half_t acc_s_cast[16];
  float scores_sum[2];
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    *(uint4*)(((half_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 31) >> 3) * 4096) + (i * 256)) + ((((int)threadIdx.x) >> 5) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)) = *(uint4*)(Q + ((((((((int)blockIdx.z) * 6291456) + (((int)blockIdx.x) * 196608)) + (i * 12288)) + ((((int)threadIdx.x) >> 5) * 3072)) + (((int)blockIdx.y) * 256)) + ((((int)threadIdx.x) & 31) * 8)));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 128; ++i_1) {
    acc_o[i_1] = 0.000000e+00f;
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 2; ++i_2) {
    logsum[i_2] = 0.000000e+00f;
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 2; ++i_3) {
    scores_max[i_3] = -CUDART_INF_F;
  }
  __syncthreads();
  #pragma unroll
  for (int i_4 = 0; i_4 < 64; ++i_4) {
    *(uint1*)(Q_local + (i_4 * 2)) = *(uint1*)(((half_t*)buf_dyn_shmem) + ((((((((((i_4 >> 4) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + ((i_4 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((i_4 & 15) >> 3)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_4 & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_4 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 8192));
  }
  #pragma unroll
  for (int i_5 = 0; i_5 < 128; ++i_5) {
    Q_local[i_5] = ((half_t)(((float)Q_local[i_5]) * 9.016844e-02f));
  }
  __syncthreads();
  #pragma unroll
  for (int i_6 = 0; i_6 < 8; ++i_6) {
    tl::cp_async_gs<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 31) >> 3) * 4096) + (i_6 * 512)) + ((((int)threadIdx.x) >> 5) * 128)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_6 & 1)) & 1) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 16384), K+(((((((int)blockIdx.z) * 6291456) + (i_6 * 12288)) + ((((int)threadIdx.x) >> 5) * 3072)) + (((int)blockIdx.y) * 256)) + ((((int)threadIdx.x) & 31) * 8)));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_7 = 0; i_7 < 8; ++i_7) {
    tl::cp_async_gs<16>(buf_dyn_shmem+((((((((((int)threadIdx.x) & 31) >> 3) * 4096) + (i_7 * 512)) + ((((int)threadIdx.x) >> 5) * 128)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_7 & 1)) & 1) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 16)), V+(((((((int)blockIdx.z) * 6291456) + (i_7 * 12288)) + ((((int)threadIdx.x) >> 5) * 3072)) + (((int)blockIdx.y) * 256)) + ((((int)threadIdx.x) & 31) * 8)));
  }
  tl::cp_async_commit();
  for (int k = 0; k < ((((int)blockIdx.x) * 2) + 1); ++k) {
    #pragma unroll
    for (int i_8 = 0; i_8 < 16; ++i_8) {
      acc_s[i_8] = ((((((k * 32) + ((i_8 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_8 & 1)) <= ((((((int)blockIdx.x) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_8 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))) ? 0.000000e+00f : -CUDART_INF_F);
    }
    tl::cp_async_wait<1>();
    __syncthreads();
    tl::gemm_rs<64, 32, 256, 4, 1, 0, 1>((&(Q_local[0])), (&(((half_t*)buf_dyn_shmem)[8192])), (&(acc_s[0])));
    __syncthreads();
    #pragma unroll
    for (int i_9 = 0; i_9 < 8; ++i_9) {
      tl::cp_async_gs<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 31) >> 3) * 4096) + (i_9 * 512)) + ((((int)threadIdx.x) >> 5) * 128)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_9 & 1)) & 1) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 16384), K+(((((((((int)blockIdx.z) * 6291456) + (k * 98304)) + (i_9 * 12288)) + ((((int)threadIdx.x) >> 5) * 3072)) + (((int)blockIdx.y) * 256)) + ((((int)threadIdx.x) & 31) * 8)) + 98304));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_10 = 0; i_10 < 2; ++i_10) {
      scores_max_prev[i_10] = scores_max[i_10];
    }
    #pragma unroll
    for (int i_11 = 0; i_11 < 2; ++i_11) {
      #pragma unroll
      for (int rv = 0; rv < 8; ++rv) {
        scores_max[i_11] = max(scores_max[i_11], acc_s[((((rv & 3) * 4) + (i_11 * 2)) + (rv >> 2))]);
      }
      scores_max[i_11] = tl::AllReduce<tl::MaxOp, 4, 1>::run(scores_max[i_11]);
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 2; ++i_12) {
      scores_scale[i_12] = exp2f((scores_max_prev[i_12] - scores_max[i_12]));
    }
    #pragma unroll
    for (int i_13 = 0; i_13 < 128; ++i_13) {
      acc_o[i_13] = (acc_o[i_13] * scores_scale[((i_13 & 3) >> 1)]);
    }
    #pragma unroll
    for (int i_14 = 0; i_14 < 16; ++i_14) {
      acc_s[i_14] = exp2f((acc_s[i_14] - scores_max[((i_14 & 3) >> 1)]));
    }
    #pragma unroll
    for (int i_15 = 0; i_15 < 16; ++i_15) {
      acc_s_cast[i_15] = ((half_t)acc_s[i_15]);
    }
    tl::cp_async_wait<1>();
    __syncthreads();
    tl::gemm_rs<64, 256, 32, 4, 1, 0, 0>((&(acc_s_cast[0])), (&(((half_t*)buf_dyn_shmem)[0])), (&(acc_o[0])));
    __syncthreads();
    #pragma unroll
    for (int i_16 = 0; i_16 < 8; ++i_16) {
      tl::cp_async_gs<16>(buf_dyn_shmem+((((((((((int)threadIdx.x) & 31) >> 3) * 4096) + (i_16 * 512)) + ((((int)threadIdx.x) >> 5) * 128)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_16 & 1)) & 1) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 16)), V+(((((((((int)blockIdx.z) * 6291456) + (k * 98304)) + (i_16 * 12288)) + ((((int)threadIdx.x) >> 5) * 3072)) + (((int)blockIdx.y) * 256)) + ((((int)threadIdx.x) & 31) * 8)) + 98304));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_17 = 0; i_17 < 2; ++i_17) {
      scores_sum[i_17] = 0.000000e+00f;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 8; ++rv_1) {
        scores_sum[i_17] = (scores_sum[i_17] + acc_s[((((rv_1 & 3) * 4) + (i_17 * 2)) + (rv_1 >> 2))]);
      }
      scores_sum[i_17] = tl::AllReduce<tl::SumOp, 4, 1>::run(scores_sum[i_17]);
    }
    #pragma unroll
    for (int i_18 = 0; i_18 < 2; ++i_18) {
      logsum[i_18] = ((logsum[i_18] * scores_scale[i_18]) + scores_sum[i_18]);
    }
  }
  #pragma unroll
  for (int i_19 = 0; i_19 < 16; ++i_19) {
    acc_s[i_19] = (((((((((int)blockIdx.x) * 64) + ((i_19 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_19 & 1)) + 32) <= ((((((int)blockIdx.x) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_19 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))) ? 0.000000e+00f : -CUDART_INF_F);
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  tl::gemm_rs<64, 32, 256, 4, 1, 0, 1>((&(Q_local[0])), (&(((half_t*)buf_dyn_shmem)[8192])), (&(acc_s[0])));
  #pragma unroll
  for (int i_20 = 0; i_20 < 2; ++i_20) {
    scores_max_prev[i_20] = scores_max[i_20];
  }
  #pragma unroll
  for (int i_21 = 0; i_21 < 2; ++i_21) {
    #pragma unroll
    for (int rv_2 = 0; rv_2 < 8; ++rv_2) {
      scores_max[i_21] = max(scores_max[i_21], acc_s[((((rv_2 & 3) * 4) + (i_21 * 2)) + (rv_2 >> 2))]);
    }
    scores_max[i_21] = tl::AllReduce<tl::MaxOp, 4, 1>::run(scores_max[i_21]);
  }
  #pragma unroll
  for (int i_22 = 0; i_22 < 2; ++i_22) {
    scores_scale[i_22] = exp2f((scores_max_prev[i_22] - scores_max[i_22]));
  }
  #pragma unroll
  for (int i_23 = 0; i_23 < 128; ++i_23) {
    acc_o[i_23] = (acc_o[i_23] * scores_scale[((i_23 & 3) >> 1)]);
  }
  #pragma unroll
  for (int i_24 = 0; i_24 < 16; ++i_24) {
    acc_s[i_24] = exp2f((acc_s[i_24] - scores_max[((i_24 & 3) >> 1)]));
  }
  #pragma unroll
  for (int i_25 = 0; i_25 < 16; ++i_25) {
    acc_s_cast[i_25] = ((half_t)acc_s[i_25]);
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  tl::gemm_rs<64, 256, 32, 4, 1, 0, 0>((&(acc_s_cast[0])), (&(((half_t*)buf_dyn_shmem)[0])), (&(acc_o[0])));
  #pragma unroll
  for (int i_26 = 0; i_26 < 2; ++i_26) {
    scores_sum[i_26] = 0.000000e+00f;
    #pragma unroll
    for (int rv_3 = 0; rv_3 < 8; ++rv_3) {
      scores_sum[i_26] = (scores_sum[i_26] + acc_s[((((rv_3 & 3) * 4) + (i_26 * 2)) + (rv_3 >> 2))]);
    }
    scores_sum[i_26] = tl::AllReduce<tl::SumOp, 4, 1>::run(scores_sum[i_26]);
  }
  #pragma unroll
  for (int i_27 = 0; i_27 < 2; ++i_27) {
    logsum[i_27] = ((logsum[i_27] * scores_scale[i_27]) + scores_sum[i_27]);
  }
  #pragma unroll
  for (int i_28 = 0; i_28 < 128; ++i_28) {
    acc_o[i_28] = (acc_o[i_28] / logsum[((i_28 & 3) >> 1)]);
  }
  #pragma unroll
  for (int i_29 = 0; i_29 < 64; ++i_29) {
    uint1 __1;
    float2 v_ = *(float2*)(acc_o + (i_29 * 2));
    ((half2*)(&(__1.x)))->x = (half_t)(v_.x);
    ((half2*)(&(__1.x)))->y = (half_t)(v_.y);
    *(uint1*)(Output + ((((((((((int)blockIdx.z) * 6291456) + (((int)blockIdx.x) * 196608)) + ((((int)threadIdx.x) >> 5) * 49152)) + ((i_29 & 1) * 24576)) + (((((int)threadIdx.x) & 31) >> 2) * 3072)) + (((int)blockIdx.y) * 256)) + ((i_29 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __1;
  }
}
