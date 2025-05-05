/* utils/fake_libc_include/simd_cuda.h */
#ifndef SIMD_CUDA_H
#define SIMD_CUDA_H

/* 1) half */
/* Use a 16‑bit integer to stand in for __fp16 */
typedef unsigned short half;

/* 2) CUDA built‑ins */
struct dim3 { int x, y, z; };
extern struct dim3 blockIdx;
extern struct dim3 threadIdx;

/* 3) WMMA fragment type (no templates, no namespace) */
typedef struct { float x; } wmma_fragment;

/* 4) WMMA‑style API as plain C functions */
void wmma_fill_fragment(wmma_fragment frag, float val);
void wmma_load_matrix_sync(wmma_fragment frag, const half *ptr, int stride);
void wmma_mma_sync(wmma_fragment dst,
                   wmma_fragment a,
                   wmma_fragment b,
                   wmma_fragment c);
void wmma_store_matrix_sync(float *ptr,
                            wmma_fragment frag,
                            int stride,
                            int layout);

/* 5) Layout constants */
#define wmma_mem_row_major 0

#endif /* SIMD_CUDA_H */