# NVIDIA GPU → CPU DL Boost

CUDA_TO_CPU_PROMPT = """
You are an expert in low-level deep learning compiler optimization.

Task:
Translate the following CUDA kernel into optimized CPU code using AVX VNNI intrinsics if possible.

Constraints:
- Match the numerical accuracy.
- Try to preserve parallelism using SIMD (e.g., AVX VNNI).
- Use intrinsics instead of OpenMP or naive loops.
- Keep code complete with includes, main function, and comments.

Input CUDA code:
```cpp
{input_code}
```
Now generate the equivalent optimized CPU code:
"""

# 2. NVIDIA GPU → AMD GPU
CUDA_TO_AMD_PROMPT = """
You are an expert in GPU kernel development.

Task:
Translate the following CUDA kernel into HIP kernel code optimized for AMD GPUs.

Constraints:
- Ensure correctness and numerical accuracy.
- Preserve thread/block-level parallelism.
- Use HIP-specific APIs and syntax.
- Include necessary headers and kernel launch code.

Input CUDA code:
```cpp
{input_code}
```
Now generate the equivalent HIP kernel for AMD GPU: """

# 3. NVIDIA GPU → MLU
CUDA_TO_MLU_PROMPT = """
You are a domain expert in NPU programming.

Task:
Translate the following CUDA kernel into MLU kernel code optimized for SIMD-style NPU execution.

Constraints:
- Keep numerical accuracy.
- Match memory layout and thread-level parallelism with MLU cores.
- Replace CUDA-specific APIs with MLU equivalents.
- Keep code complete with all necessary includes and launch annotations.

Input CUDA code:
```cpp
{input_code}
```
Now generate the equivalent MLU kernel code: """

# 4. AMD GPU → CPU DL Boost
HIP_TO_CPU_PROMPT = """
You are an expert in compiler optimization.

Task:
Convert the following HIP (AMD GPU) kernel into optimized CPU code using AVX VNNI intrinsics if applicable.

Constraints:
- Preserve numerical accuracy and data layout.
- Use SIMD intrinsics rather than simple scalar loops.
- Include full C++ code with headers, intrinsics, and comments.

Input HIP code:
```cpp
{input_code}
```
Now generate the equivalent MLU kernel code: """

# 5. AMD GPU → NVIDIA GPU
HIP_TO_CUDA_PROMPT = """
You are a GPU programming expert.

Task:
Convert the following HIP (AMD GPU) kernel into equivalent CUDA kernel code for NVIDIA GPUs.

Constraints:
- Preserve thread hierarchy and parallelism.
- Match behavior and performance as much as possible.
- Replace HIP API with appropriate CUDA API.
- Include complete kernel function and launch setup.

Input HIP code:
```cpp
{input_code}
```
Now generate the equivalent MLU kernel code: """

# 6. AMD GPU → MLU

HIP_TO_MLU_PROMPT = """
You are an expert in heterogeneous deep learning compilers.

Task:
Convert the following HIP kernel into MLU code for execution on SIMD-based NPU architecture.

Constraints:
- Preserve the logic and numerical accuracy.
- Use MLU's parallel execution model.
- Rewrite HIP memory and kernel APIs using MLU equivalents.

Input HIP code:
```cpp
{input_code}
```
Now generate the equivalent MLU kernel code: """

# 7. CPU DL Boost → NVIDIA GPU
CPU_TO_CUDA_PROMPT = """
You are a high-performance code generation expert.

Task:
Convert the following CPU code (with AVX intrinsics or scalar operations) into a CUDA kernel optimized for NVIDIA GPUs.

Constraints:
- Exploit thread-level parallelism using CUDA.
- Ensure correctness and similar performance characteristics.
- Include complete kernel function and necessary CUDA headers.

Input CPU code:
```cpp
{input_code}
```
Now generate the equivalent MLU kernel code: """

# 8. CPU DL Boost → AMD GPU
CPU_TO_HIP_PROMPT = """
You are an expert in heterogeneous programming.

Task:
Translate the following AVX-accelerated CPU code into a HIP kernel targeting AMD GPUs.

Constraints:
- Match data layout and computation logic.
- Use HIP kernel launch structure.
- Preserve performance via parallel threads and memory coalescing.

Input CPU code:
```cpp
{input_code}
```
Now generate the equivalent MLU kernel code: """

# 9. CPU DL Boost → MLU

CPU_TO_MLU_PROMPT = """
You are an expert in NPU compilation and optimization.

Task:
Convert the following CPU-based computation (using scalar or SIMD intrinsics) into an MLU-compatible kernel.

Constraints:
- Translate vectorized CPU logic into MLU's SIMD execution model.
- Ensure correctness and performance.
- Include all necessary MLU intrinsics and kernel launch code.

Input CPU code:
```cpp
{input_code}
```
Now generate the equivalent MLU kernel code: """

# 10. MLU → CPU DL Boost
MLU_TO_CPU_PROMPT = """
You are an expert in code portability and SIMD optimization.

Task:
Convert the following MLU kernel into optimized CPU code using AVX VNNI intrinsics.

Constraints:
- Preserve correctness and layout.
- Replace MLU SIMD intrinsics with AVX equivalents.
- Provide complete and compilable C++ code.

Input MLU code:
```cpp
{input_code}
```
Now generate the equivalent MLU kernel code: """

# 11. MLU → NVIDIA GPU
MLU_TO_CUDA_PROMPT = """
You are a deep learning compiler expert.

Task:
Convert the following MLU kernel (SIMD-based) into an equivalent CUDA kernel optimized for NVIDIA GPUs.

Constraints:
- Replace MLU intrinsics with CUDA thread/block operations.
- Preserve memory access logic and computation layout.
- Include full kernel and proper annotations.

Input MLU code:
```cpp
{input_code}
```
Now generate the equivalent MLU kernel code: """

# 12. MLU → AMD GPU
MLU_TO_HIP_PROMPT = """
You are a heterogeneous systems engineer.

Task:
Translate the following MLU kernel into a HIP kernel targeting AMD GPUs.

Constraints:
- Match performance and layout.
- Use HIP thread/block parallelism to simulate MLU SIMD lanes.
- Include full HIP kernel and supporting launch code.

Input MLU code:
```cpp
{input_code}
```
Now generate the equivalent MLU kernel code: """
