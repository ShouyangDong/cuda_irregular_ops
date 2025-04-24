# NVIDIA GPU → CPU DL Boost

CUDA_TO_CPU_PROMPT = """
You are an expert in low-level deep learning compiler optimization.

## Task:
Translate the following CUDA kernel into optimized CPU code for AVX VNNI or scalar instructions.

## Constraints:
- Maintain numerical correctness.
- Target integer operations (e.g., int8, uint8 with dot-product accumulation).
- Use AVX VNNI intrinsics (like `_mm512_dpbusd_epi32`) or scalar integer code when applicable.
- Avoid floating point AVX (e.g., `_mm256_add_ps`) unless explicitly required.
- Avoid OpenMP or thread-level parallelism.
- Keep code self-contained with includes and comments.

---

### Example 1

**Input CUDA code:**
```cpp
extern "C" __global__ void dot_add(const uint8_t* A, const int8_t* B, int* C) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < 16) {
    int acc = 0;
    for (int j = 0; j < 64; ++j) {
      acc += A[i * 64 + j] * B[i * 64 + j];
    }
    C[i] = acc;
  }
}
```

**Output CPU code (with AVX VNNI):**
```cpp
#include <immintrin.h>
#include <stdint.h>
#include <string.h>

extern "C" void dot_add(const uint8_t* A, const int8_t* B, int* C) {
  for (int i = 0; i < 16; ++i) {
    __m512i acc = _mm512_setzero_si512();
    for (int j = 0; j < 64; j += 64) {
      __m512i va = _mm512_loadu_si512((__m512i*)(A + i * 64 + j));
      __m512i vb = _mm512_loadu_si512((__m512i*)(B + i * 64 + j));
      acc = _mm512_dpbusd_epi32(acc, va, vb);
    }
    _mm512_storeu_si512((__m512i*)&C[i], acc); // or scalar extract + store
  }
}
```
---

### Example 2

**Input CUDA code:**
```cpp
extern "C" __global__ void add_kernel(const int* A, const int* B, int* C) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < 32) {
    C[idx] = A[idx] + B[idx];
  }
}
```

**Output CPU code (with scalar ops):**
```cpp
#include <stdint.h>

extern "C" void add_kernel(const int* A, const int* B, int* C) {
  for (int i = 0; i < 32; ++i) {
    C[i] = A[i] + B[i];
  }
}
```

---

Now, translate the following CUDA code into AVX VNNI or scalar CPU code:

```cpp
{input_code}
```

Generate the complete and optimized CPU code:
"""

# NVIDIA GPU → AMD GPU (HIP)

CUDA_TO_AMD_PROMPT = """
You are an expert in GPU compiler optimization and cross-platform GPU kernel translation.

## Task:
Translate the following CUDA kernel into HIP for AMD GPUs.

## Constraints:
- Ensure functional correctness and maintain the same parallel structure.
- Convert CUDA-specific APIs (e.g., thread/block indexing, memory access, intrinsics) to their HIP equivalents.
- Ensure all headers and kernel launch configurations are valid under HIP.
- Do not include device query or runtime setup; focus on the kernel and launch syntax.

---

### Example 1

**Input CUDA code:**
```cpp
extern "C" __global__ void add_kernel(const float* A, const float* B, float* C) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < 1024) {
    C[idx] = A[idx] + B[idx];
  }
}
```

**Output HIP code:**
```cpp
#include <hip/hip_runtime.h>

extern "C" __global__ void add_kernel(const float* A, const float* B, float* C) {
  int idx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  if (idx < 1024) {
    C[idx] = A[idx] + B[idx];
  }
}
```

---

### Example 2

**Input CUDA code:**
```cpp
extern "C" __global__ void scale(const float* A, float* B, float alpha) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < 256) {
    B[tid] = alpha * A[tid];
  }
}
```

**Output HIP code:**
```cpp
#include <hip/hip_runtime.h>

extern "C" __global__ void scale(const float* A, float* B, float alpha) {
  int tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  if (tid < 256) {
    B[tid] = alpha * A[tid];
  }
}
```

---

Now, translate the following CUDA code into HIP:

```cpp
{input_code}
```

Generate the complete and correct HIP kernel:
"""

# AMD GPU → CPU DL Boost

HIP_TO_CPU_PROMPT = """
You are an expert in low-level deep learning compiler optimization.

## Task:
Translate the following HIP kernel for AMD GPU into optimized CPU code using AVX VNNI or scalar instructions.

## Constraints:
- Maintain numerical correctness.
- Focus on integer operations when possible (e.g., int8/uint8 with dot-product accumulation).
- Use AVX VNNI intrinsics like `_mm512_dpbusd_epi32` or scalar integer code.
- Avoid floating point AVX intrinsics (e.g., `_mm256_add_ps`) unless required.
- Avoid OpenMP or thread-level parallelism.
- Keep code self-contained with includes and comments.

---

### Example 1

**Input HIP code:**
```cpp
#include <hip/hip_runtime.h>

extern "C" __global__ void dot_add(const uint8_t* A, const int8_t* B, int* C) {
  int i = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  if (i < 16) {
    int acc = 0;
    for (int j = 0; j < 64; ++j) {
      acc += A[i * 64 + j] * B[i * 64 + j];
    }
    C[i] = acc;
  }
}
```

**Output CPU code (with AVX VNNI):**
```cpp
#include <immintrin.h>
#include <stdint.h>
#include <string.h>

extern "C" void dot_add(const uint8_t* A, const int8_t* B, int* C) {
  for (int i = 0; i < 16; ++i) {
    __m512i acc = _mm512_setzero_si512();
    for (int j = 0; j < 64; j += 64) {
      __m512i va = _mm512_loadu_si512((__m512i*)(A + i * 64 + j));
      __m512i vb = _mm512_loadu_si512((__m512i*)(B + i * 64 + j));
      acc = _mm512_dpbusd_epi32(acc, va, vb);
    }
    _mm512_storeu_si512((__m512i*)&C[i], acc); // optional scalar store
  }
}
```

---

### Example 2

**Input HIP code:**
```cpp
#include <hip/hip_runtime.h>

extern "C" __global__ void add_kernel(const int* A, const int* B, int* C) {
  int idx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  if (idx < 32) {
    C[idx] = A[idx] + B[idx];
  }
}
```

**Output CPU code (scalar):**
```cpp
#include <stdint.h>

extern "C" void add_kernel(const int* A, const int* B, int* C) {
  for (int i = 0; i < 32; ++i) {
    C[i] = A[i] + B[i];
  }
}
```

---

Now, translate the following HIP kernel into optimized CPU code using AVX VNNI or scalar instructions:

```cpp
{input_code}
```

Generate the complete and optimized CPU implementation:
"""

# AMD GPU → NVIDIA GPU

HIP_TO_CUDA_PROMPT = """
You are an expert in GPU kernel development and cross-platform code translation.

## Task:
Translate the following HIP kernel code (AMD GPU) into CUDA kernel code (NVIDIA GPU).

## Constraints:
- Match functionality and numerical correctness.
- Translate HIP-specific APIs (e.g., `hipThreadIdx_x`) into their CUDA equivalents (e.g., `threadIdx.x`).
- Replace HIP headers with CUDA headers.
- Maintain the kernel launch structure and semantics.
- Output self-contained, compilable CUDA code with includes and comments.

---

### Example 1

**Input HIP code:**
```cpp
#include <hip/hip_runtime.h>

extern "C" __global__ void vector_add(const float* A, const float* B, float* C, int N) {
  int idx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  if (idx < N) {
    C[idx] = A[idx] + B[idx];
  }
}
```

**Output CUDA code:**
```cpp
#include <cuda_runtime.h>

extern "C" __global__ void vector_add(const float* A, const float* B, float* C, int N) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N) {
    C[idx] = A[idx] + B[idx];
  }
}
```

---

### Example 2

**Input HIP code:**
```cpp
#include <hip/hip_runtime.h>

__global__ void scale_kernel(float* data, float scale, int size) {
  int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  if (idx < size) {
    data[idx] *= scale;
  }
}
```

**Output CUDA code:**
```cpp
#include <cuda_runtime.h>

__global__ void scale_kernel(float* data, float scale, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    data[idx] *= scale;
  }
}
```

---

Now, translate the following HIP kernel into CUDA kernel code:

```cpp
{input_code}
```

Generate the complete and correct CUDA kernel code:
"""

# CPU DLBoost → NVIDIA GPU

CPU_TO_CUDA_PROMPT = """
You are an expert in GPU kernel development and compiler backend translation.

## Task:
Translate the following CPU kernel code (using scalar operations or AVX VNNI intrinsics) into CUDA kernel code for NVIDIA GPUs.

## Constraints:
- Match functionality and numerical accuracy.
- Utilize GPU parallelism using threads (`threadIdx`, `blockIdx`) and avoid scalar-only code.
- Do not use AVX or CPU-specific intrinsics.
- Ensure the CUDA code is self-contained and ready to compile, with includes and comments.

---

### Example 1

**Input CPU code (scalar):**
```cpp
#include <stdint.h>

extern "C" void add_kernel(const int* A, const int* B, int* C) {
  for (int i = 0; i < 32; ++i) {
    C[i] = A[i] + B[i];
  }
}
```

**Output CUDA code:**
```cpp
#include <cuda_runtime.h>

__global__ void add_kernel(const int* A, const int* B, int* C) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < 32) {
    C[idx] = A[idx] + B[idx];
  }
}
```

---

### Example 2

**Input CPU code (AVX VNNI):**
```cpp
#include <immintrin.h>
#include <stdint.h>

extern "C" void dot_add(const uint8_t* A, const int8_t* B, int* C) {
  for (int i = 0; i < 16; ++i) {
    __m512i acc = _mm512_setzero_si512();
    for (int j = 0; j < 64; j += 64) {
      __m512i va = _mm512_loadu_si512((__m512i*)(A + i * 64 + j));
      __m512i vb = _mm512_loadu_si512((__m512i*)(B + i * 64 + j));
      acc = _mm512_dpbusd_epi32(acc, va, vb);
    }
    C[i] = ((int*)&acc)[0]; // simplified scalar store
  }
}
```

**Output CUDA code:**
```cpp
#include <cuda_runtime.h>

__global__ void dot_add(const uint8_t* A, const int8_t* B, int* C) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < 16) {
    int acc = 0;
    for (int j = 0; j < 64; ++j) {
      acc += A[i * 64 + j] * B[i * 64 + j];
    }
    C[i] = acc;
  }
}
```

---

Now, translate the following DLBoost CPU code into a correct and complete CUDA kernel:

```cpp
{input_code}
```

Generate the full CUDA implementation:
"""

# CPU DLBoost → AMD GPU (HIP)

CPU_TO_HIP_PROMPT = """
You are an expert in GPU kernel optimization and system-level code translation.

## Task:
Translate the following CPU kernel (scalar or AVX VNNI) code into an equivalent AMD GPU kernel using HIP.

## Constraints:
- Maintain numerical correctness and functional behavior.
- Use HIP GPU thread parallelism (e.g., `hipThreadIdx_x`, `hipBlockIdx_x`, etc.).
- Avoid CPU-specific intrinsics like AVX, SSE, or VNNI.
- Output must be complete HIP code with includes and comments.

---

### Example 1

**Input CPU code (scalar):**
```cpp
#include <stdint.h>

extern "C" void add_kernel(const int* A, const int* B, int* C) {
  for (int i = 0; i < 32; ++i) {
    C[i] = A[i] + B[i];
  }
}
```

**Output HIP code:**
```cpp
#include <hip/hip_runtime.h>

__global__ void add_kernel(const int* A, const int* B, int* C) {
  int idx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  if (idx < 32) {
    C[idx] = A[idx] + B[idx];
  }
}
```

---

### Example 2

**Input CPU code (AVX VNNI style, simplified):**
```cpp
#include <immintrin.h>
#include <stdint.h>

extern "C" void dot_add(const uint8_t* A, const int8_t* B, int* C) {
  for (int i = 0; i < 16; ++i) {
    int acc = 0;
    for (int j = 0; j < 64; ++j) {
      acc += A[i * 64 + j] * B[i * 64 + j];
    }
    C[i] = acc;
  }
}
```

**Output HIP code:**
```cpp
#include <hip/hip_runtime.h>
#include <stdint.h>

__global__ void dot_add(const uint8_t* A, const int8_t* B, int* C) {
  int i = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  if (i < 16) {
    int acc = 0;
    for (int j = 0; j < 64; ++j) {
      acc += A[i * 64 + j] * B[i * 64 + j];
    }
    C[i] = acc;
  }
}
```

---

Now translate the following CPU DLBoost code into an equivalent HIP kernel for AMD GPUs:

```cpp
{input_code}
```

Generate the full HIP implementation:
"""

# **MLU_TO_CPU_PROMPT**

MLU_TO_CPU_PROMPT = """
You are an expert in low-level deep learning compiler optimization.

## Task:
Translate the following MLU (BANG C) kernel into optimized CPU code using AVX VNNI intrinsics or scalar code when appropriate.

## Constraints:
- Match the numerical correctness.
- Use AVX VNNI intrinsics for int8 dot-product accumulation if applicable.
- Use scalar code for float32 computation (avoid AVX `_mm256_add_ps`, etc.).
- Do not use OpenMP or thread-level parallelism.
- Eliminate MLU-specific constructs like `__memcpy`, `__nram__`, and `__bang_*`.
- Keep the output self-contained with includes, comments, and `extern "C"`.

---

### Example 1

**Input MLU code:**
```cpp
extern "C" __mlu_global__ void add(float *lhs, float *rhs, float *add_1935) {
  __nram__ float lhs_local_nram[2048];
  __memcpy(((float *)lhs_local_nram + (0)),
           ((float *)lhs + ((((int)coreId) * 1024))), 4096, GDRAM2NRAM);
  __memcpy(((float *)lhs_local_nram + (1024)),
           ((float *)rhs + ((((int)coreId) * 1024))), 4096, GDRAM2NRAM);
  __bang_add(((float *)lhs_local_nram + (0)), ((float *)lhs_local_nram + (0)),
             ((float *)lhs_local_nram + (1024)), 1024);
  __memcpy(((float *)add_1935 + ((((int)coreId) * 1024))),
           ((float *)lhs_local_nram + (0)), 4096, NRAM2GDRAM);
}
```

**Output CPU code (scalar):**
```cpp
#include <stdint.h>

extern "C" void add(float *lhs, float *rhs, float *add_1935) {
  for (int coreId = 0; coreId < 4; ++coreId) {
    int base = coreId * 1024;
    for (int i = 0; i < 1024; ++i) {
      add_1935[base + i] = lhs[base + i] + rhs[base + i];
    }
  }
}
```

---

### Example 2

**Input MLU code:**
```cpp
extern "C" __mlu_global__ void sub(float *lhs, float *rhs, float *output) {
  __nram__ float local_buf[2048];
  __memcpy(local_buf, lhs + coreId * 1024, 4096, GDRAM2NRAM);
  __memcpy(local_buf + 1024, rhs + coreId * 1024, 4096, GDRAM2NRAM);
  __bang_sub(local_buf, local_buf, local_buf + 1024, 1024);
  __memcpy(output + coreId * 1024, local_buf, 4096, NRAM2GDRAM);
}
```

**Output CPU code (scalar):**
```cpp
#include <stdint.h>

extern "C" void sub(float *lhs, float *rhs, float *output) {
  for (int coreId = 0; coreId < 4; ++coreId) {
    int base = coreId * 1024;
    for (int i = 0; i < 1024; ++i) {
      output[base + i] = lhs[base + i] - rhs[base + i];
    }
  }
}
```

---

Now translate the following MLU kernel into optimized CPU code:
```cpp
{input_code}
```

Generate the complete and optimized CPU code:
"""
# **CPU_TO_MLU_PROMPT**

CPU_TO_MLU_PROMPT = """
You are an expert in low-level deep learning compiler optimization.

## Task:
Translate the following CPU C/C++ code into optimized MLU (BANG C) code.

## Constraints:
- Match the numerical correctness.
- Exploit MLU hardware features like NRAM memory and BANG instructions.
- Use `__memcpy` for explicit data movement between GDRAM and NRAM.
- Use `__bang_*` intrinsics for vector computation.
- Parallelize across `coreId` (typically 0–3 or 0–4), assuming 4 cores.
- Wrap the kernel in `extern "C" __mlu_global__ void`.
- Keep the output self-contained with comments and proper buffer sizes.

---

### Example 1

**Input CPU code:**
```cpp
extern "C" void add(float *lhs, float *rhs, float *add_1935) {
  for (int coreId = 0; coreId < 4; ++coreId) {
    int base = coreId * 1024;
    for (int i = 0; i < 1024; ++i) {
      add_1935[base + i] = lhs[base + i] + rhs[base + i];
    }
  }
}
```

**Output MLU code:**
```cpp
extern "C" __mlu_global__ void add(float *lhs, float *rhs, float *add_1935) {
  __nram__ float lhs_local_nram[2048];

  // Load lhs and rhs from GDRAM to NRAM
  __memcpy(lhs_local_nram,
           lhs + (coreId * 1024),
           1024 * sizeof(float),
           GDRAM2NRAM);
  __memcpy(lhs_local_nram + 1024,
           rhs + (coreId * 1024),
           1024 * sizeof(float),
           GDRAM2NRAM);

  // Compute addition using __bang_add
  __bang_add(lhs_local_nram,
             lhs_local_nram,
             lhs_local_nram + 1024,
             1024);

  // Write result back to GDRAM
  __memcpy(add_1935 + (coreId * 1024),
           lhs_local_nram,
           1024 * sizeof(float),
           NRAM2GDRAM);
}
```

---

### Example 2

**Input CPU code:**
```cpp
extern "C" void relu(float* input, float* output) {
  for (int coreId = 0; coreId < 4; ++coreId) {
    int base = coreId * 512;
    for (int i = 0; i < 512; ++i) {
      output[base + i] = input[base + i] > 0 ? input[base + i] : 0;
    }
  }
}
```

**Output MLU code:**
```cpp
extern "C" __mlu_global__ void relu(float* input, float* output) {
  __nram__ float buf[512];

  __memcpy(buf, input + coreId * 512, 512 * sizeof(float), GDRAM2NRAM);
  __bang_relu(buf, buf, 512);
  __memcpy(output + coreId * 512, buf, 512 * sizeof(float), NRAM2GDRAM);
}
```

---

Now translate the following CPU code into MLU BANG C kernel code:
```cpp
{input_code}
```

Generate the complete and optimized MLU code:
"""

# **MLU_TO_CUDA_PROMPT**


MLU_TO_CUDA_PROMPT = """
You are an expert in low-level deep learning compiler optimization.

## Task:
Translate the following MLU (BANG C) kernel into equivalent CUDA kernel code.

## Constraints:
- Match the numerical correctness.
- Remove `__nram__` buffers and use global memory directly or shared memory if necessary.
- Replace `__memcpy` with direct memory access or `memcpy` if appropriate.
- Replace `__bang_*` operations with CUDA equivalents or scalar loops.
- Wrap the kernel in `extern "C" __global__`.
- Use `threadIdx.x`, `blockIdx.x`, etc., to emulate `coreId`-style parallelism.
- Provide complete, compilable CUDA code with comments.

---

### Example 1

**Input MLU code:**
```cpp
extern "C" __mlu_global__ void add(float *lhs, float *rhs, float *add_1935) {
  __nram__ float lhs_local_nram[2048];

  __memcpy(lhs_local_nram,
           lhs + (coreId * 1024),
           1024 * sizeof(float),
           GDRAM2NRAM);
  __memcpy(lhs_local_nram + 1024,
           rhs + (coreId * 1024),
           1024 * sizeof(float),
           GDRAM2NRAM);

  __bang_add(lhs_local_nram,
             lhs_local_nram,
             lhs_local_nram + 1024,
             1024);

  __memcpy(add_1935 + (coreId * 1024),
           lhs_local_nram,
           1024 * sizeof(float),
           NRAM2GDRAM);
}
```

**Output CUDA code:**
```cpp
extern "C" __global__ void add(float *lhs, float *rhs, float *add_1935) {
  int coreId = threadIdx.x + blockIdx.x * blockDim.x;

  if (coreId < 4) {
    int base = coreId * 1024;

    for (int i = 0; i < 1024; ++i) {
      add_1935[base + i] = lhs[base + i] + rhs[base + i];
    }
  }
}
```

---

### Example 2

**Input MLU code:**
```cpp
extern "C" __mlu_global__ void relu(float* input, float* output) {
  __nram__ float buf[512];

  __memcpy(buf, input + coreId * 512, 512 * sizeof(float), GDRAM2NRAM);
  __bang_relu(buf, buf, 512);
  __memcpy(output + coreId * 512, buf, 512 * sizeof(float), NRAM2GDRAM);
}
```

**Output CUDA code:**
```cpp
extern "C" __global__ void relu(float* input, float* output) {
  int coreId = threadIdx.x + blockIdx.x * blockDim.x;

  if (coreId < 4) {
    int base = coreId * 512;

    for (int i = 0; i < 512; ++i) {
      output[base + i] = input[base + i] > 0 ? input[base + i] : 0;
    }
  }
}
```

---

Now translate the following MLU kernel into equivalent CUDA code:

```cpp
{input_code}
```

Generate the complete and optimized CUDA code:
"""

# **CUDA_TO_MLU_PROMPT**

CUDA_TO_MLU_PROMPT = """
You are an expert in low-level deep learning compiler optimization.

## Task:
Translate the following CUDA kernel into equivalent MLU (BANG C) kernel code.

## Constraints:
- Match the numerical correctness.
- Replace direct memory accesses with `__nram__` and `__memcpy` where appropriate.
- Use `__bang_*` operations for high-performance math operations (e.g., `__bang_add`).
- Use `__memcpy` for memory transfer between global and local memory.
- Parallelism should be emulated using `coreId` or other relevant constructs.
- Keep the kernel complete, compilable, and with comments.

---

### Example 1

**Input CUDA code:**
```cpp
extern "C" __global__ void add(float *lhs, float *rhs, float *add_1935) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < 1024) {
    add_1935[idx] = lhs[idx] + rhs[idx];
  }
}
```

**Output MLU code:**
```cpp
extern "C" __mlu_global__ void add(float *lhs, float *rhs, float *add_1935) {
  __nram__ float lhs_local_nram[1024];
  __nram__ float rhs_local_nram[1024];

  __memcpy(lhs_local_nram, lhs + (coreId * 1024), 1024 * sizeof(float), GDRAM2NRAM);
  __memcpy(rhs_local_nram, rhs + (coreId * 1024), 1024 * sizeof(float), GDRAM2NRAM);

  __bang_add(lhs_local_nram, lhs_local_nram, rhs_local_nram, 1024);

  __memcpy(add_1935 + (coreId * 1024), lhs_local_nram, 1024 * sizeof(float), NRAM2GDRAM);
}
```

---

### Example 2

**Input CUDA code:**
```cpp
extern "C" __global__ void relu(float* input, float* output) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < 1024) {
    output[idx] = (input[idx] > 0) ? input[idx] : 0;
  }
}
```

**Output MLU code:**
```cpp
extern "C" __mlu_global__ void relu(float* input, float* output) {
  __nram__ float input_local_nram[1024];
  __nram__ float output_local_nram[1024];

  __memcpy(input_local_nram, input + (coreId * 1024), 1024 * sizeof(float), GDRAM2NRAM);

  __bang_relu(input_local_nram, output_local_nram, 1024);

  __memcpy(output + (coreId * 1024), output_local_nram, 1024 * sizeof(float), NRAM2GDRAM);
}
```

---

Now translate the following CUDA kernel into equivalent MLU code:

```cpp
{input_code}
```

Generate the complete and optimized MLU code:
"""

MLU_TO_HIP_PROMPT = """
You are an expert in heterogeneous deep learning compiler optimization.

## Task:
Translate the following MLU (BANG C) kernel into a concise HIP kernel.

## Constraints:
- Preserve numerical correctness.
- Map `coreId`-style parallelism to `hipBlockIdx_x`/`hipThreadIdx_x`.
- Eliminate `__nram__`, `__memcpy` and `__bang_*`: use direct global loads/stores and scalar loops.
- Keep the code self-contained, compilable, and well commented.

---

### Example 1: add

**Input MLU code:**
```cpp
extern "C" __mlu_global__ void add(float *lhs, float *rhs, float *out) {
  __nram__ float buf[2048];
  __memcpy(buf,       lhs + coreId*1024, 1024*sizeof(float), GDRAM2NRAM);
  __memcpy(buf+1024,  rhs + coreId*1024, 1024*sizeof(float), GDRAM2NRAM);
  __bang_add(buf, buf, buf+1024, 1024);
  __memcpy(out + coreId*1024, buf, 1024*sizeof(float), NRAM2GDRAM);
}
```

**Output HIP code:**
```cpp

extern "C" __global__ void add(const float* lhs, const float* rhs, float* out) {
  int coreId = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  int base   = coreId * 1024;
  if (coreId < 4) {
    for (int i = 0; i < 1024; ++i) {
      out[base + i] = lhs[base + i] + rhs[base + i];
    }
  }
}
```

---

### Example 2: relu

**Input MLU code:**
```cpp
extern "C" __mlu_global__ void relu(float* in, float* out) {
  __nram__ float buf[512];
  __memcpy(buf,        in  + coreId*512, 512*sizeof(float), GDRAM2NRAM);
  __bang_relu(buf, buf, 512);
  __memcpy(out + coreId*512, buf,        512*sizeof(float), NRAM2GDRAM);
}
```

**Output HIP code:**
```cpp

extern "C" __global__ void relu(const float* in, float* out) {
  int coreId = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  int base   = coreId * 512;
  if (coreId < 4) {
    for (int i = 0; i < 512; ++i) {
      float v = in[base + i];
      out[base + i] = v > 0.0f ? v : 0.0f;
    }
  }
}
```

---

Now translate the following MLU kernel into concise HIP code:

```cpp
{input_code}
```

Generate the complete and optimized HIP kernel code:
"""

HIP_TO_MLU_PROMPT = """
You are an expert in heterogeneous deep learning compiler optimization.

## Task:
Translate the following HIP kernel into optimized MLU (BANG C) code.

## Constraints:
- Preserve numerical correctness.
- Allocate local NRAM buffers with `__nram__`.
- Use `__memcpy(..., GDRAM2NRAM)` to load from global memory into NRAM.
- Use `__bang_*` intrinsics for bulk compute (e.g., `__bang_add`, `__bang_relu`).
- Use `__memcpy(..., NRAM2GDRAM)` to write results back to global memory.
- Emulate `coreId` from `threadIdx.x` + `blockIdx.x * blockDim.x`.
- Produce a complete, compilable MLU kernel with comments.

---

### Example 1: add

**Input HIP code:**
```cpp
#include <hip/hip_runtime.h>

extern "C" __global__ void add(const float* lhs, const float* rhs, float* out) {
  int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  if (idx < 1024) {
    out[idx] = lhs[idx] + rhs[idx];
  }
}
```

**Output MLU code:**
```cpp
extern "C" __mlu_global__ void add(float* lhs, float* rhs, float* out) {
  __nram__ float buf[2048];
  __memcpy(buf,        lhs + coreId*1024, 1024*sizeof(float), GDRAM2NRAM);
  __memcpy(buf+1024,   rhs + coreId*1024, 1024*sizeof(float), GDRAM2NRAM);
  __bang_add(buf, buf, buf+1024, 1024);
  __memcpy(out + coreId*1024, buf, 1024*sizeof(float), NRAM2GDRAM);
}
```

---

### Example 2: relu

**Input HIP code:**
```cpp
#include <hip/hip_runtime.h>

extern "C" __global__ void relu(const float* in, float* out) {
  int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  if (idx < 512) {
    float v = in[idx];
    out[idx] = v > 0.0f ? v : 0.0f;
  }
}
```

**Output MLU code:**
```cpp
extern "C" __mlu_global__ void relu(float* in, float* out) {
  __nram__ float buf[512];
  __memcpy(buf,      in  + coreId*512, 512*sizeof(float), GDRAM2NRAM);
  __bang_relu(buf, buf, 512);
  __memcpy(out + coreId*512, buf, 512*sizeof(float), NRAM2GDRAM);
}
```

---

Now translate the following HIP kernel into optimized MLU code:

```cpp
{input_code}
```

Generate the complete and optimized MLU kernel:
"""
