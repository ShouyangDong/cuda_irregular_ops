DETENSORIZATION_PROMPT_BANG = """
Detensorize

Function Overview:
`DETENSORIZE` refers to the process of transforming operations that are expressed using SIMD (Single Instruction, Multiple Data) 
or vectorized instructions into scalar operations, typically implemented as sequential `for` loops. 
This conversion allows code to be more portable across different hardware architectures, 

Application Scenario:
- When targeting multiple hardware platforms with varying SIMD support, such as CPUs, GPUs, or FPGAs, `DETENSORIZE` enables developers to write a single version of the code that runs efficiently across all platforms by removing reliance on SIMD.

- SIMD instructions can sometimes obscure the logic of a program, making it harder to track down bugs. Converting SIMD operations into scalar loops via `DETENSORIZE` can improve readability and facilitate testing.

- In cases where SIMD operations do not provide a significant performance benefit or where the cost of managing vectorization is too high, `DETENSORIZE` can be used to simplify and optimize performance for scalar-based execution units.
"""

DETENSORIZATION_DEMO_BANG = """
exmaple 1:

// before:
```
__memcpy((float *)input0_local_nram + (0), (float *)input0 + (((clusterId * 256) + (coreId * 64))), 64, GDRAM2NRAM);
```

// after:
```
for (int i = 0; i < 64/sizeof(float); i++) {
    input0_local_nram[i] = input0[(((clusterId * 256) + (coreId * 64))) + i];
}
```
"""

LOOP_RECOVERY_PROMPT_CUDA = """
Loop recovery

Function Overview:
Loop recovery is designed to convert CUDA C code that utilizes GPU thread indexing (using `threadIdx`, `blockIdx`, `blockDim`, and `gridDim`) 
into standard C++ code with `for` loop structures. The goal is to remove CUDA-specific parallelism while preserving the logical flow and structure of the code.

Application Scenario:
- Use this prompt when you want to convert CUDA C code into sequential C++ code, either for environments without GPU support or to analyze/debug the logic in a CPU-based system. This transformation is useful for simplifying GPU code or porting it to CPU environments.

### Input:
A CUDA C kernel function that uses `threadIdx`, `blockIdx`, `blockDim`, and `gridDim` to define parallel execution on a GPU.

### Output:
The same logic rewritten using standard C++ for loops to emulate the behavior of thread and block indexing in a sequential CPU-based program.

### Steps for Conversion:
1. Identify the use of `blockIdx`, `threadIdx`, `blockDim`, and `gridDim` in the input CUDA code.
2. Convert the parallel structure into nested `for` loops in standard C++.
3. Replace the GPU thread index expressions with loop index variables (e.g., `blockIdx.x * blockDim.x + threadIdx.x` becomes a C++ `for` loop with the same arithmetic).
4. Ensure that all CUDA-specific syntax, such as `__global__` and `__device__`, is removed or replaced.

### GPT Task:
Transform the following CUDA C code into equivalent C++ for loop code that sequentially emulates the CUDA threading structure. The output should use nested `for` loops to replace CUDA thread indexing.
"""

LOOP_RECOVERY_DEMO_CUDA = """
Example:

#### CUDA C Code:

```cuda
__global__ void vector_add(float* A, float* B, float* C, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        C[index] = A[index] + B[index];
    }
}
```

#### Desired C++ Output:

```cpp
void vector_add(float* A, float* B, float* C, int N, int numBlocks, int threadsPerBlock) {
    // Loop over all blocks
    for (int blockIdx = 0; blockIdx < numBlocks; ++blockIdx) {
        // Loop over all threads in each block
        for (int threadIdx = 0; threadIdx < threadsPerBlock; ++threadIdx) {
            int index = blockIdx * threadsPerBlock + threadIdx;
            if (index < N) {
                C[index] = A[index] + B[index];
            }
        }
    }
}
```
"""
LOOP_RECOVERY_PROMPT_BANG = """
"""


LOOP_RECOVERY_DEMO_BANG = """
// before:
```
for (int i = 0; i < 64; i++) {
    output[coreId * 64 + i] += A[coreId * 64 + i] * 2;
}
```
// after:
```
for (int coreId = 0; coreId < 4; coreId++) {
    for (int i = 0; i < 64; i++) {
        output[coreId * 64 + i] += A[coreId * 64 + i] * 2;
    }
}
```
"""
