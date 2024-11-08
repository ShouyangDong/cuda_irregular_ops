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
5. Maximum number of threads per block is 1024, and the maximum number of blocks per grid is 256.

### GPT Task:
Transform the following CUDA C code into equivalent C++ for loop code that sequentially emulates the CUDA threading structure. The output should use nested `for` loops to replace CUDA thread indexing.
"""

LOOP_RECOVERY_DEMO_CUDA = """
Example:

#### CUDA C Code:

```cuda
extern "C" __global__ void vector_add(float* A, float* B, float* C, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        C[index] = A[index] + B[index];
    }
}
```

#### Desired C++ Output:

```cpp
void vector_add(float* A, float* B, float* C, int N) {
    // Loop over all blocks
    for (int blockIdx = 0; blockIdx < 256; ++blockIdx) {
        // Loop over all threads in each block
        for (int threadIdx = 0; threadIdx < 1024; ++threadIdx) {
            int index = blockIdx * 1024 + threadIdx;
            if (index < N) {
                C[index] = A[index] + B[index];
            }
        }
    }
}
```
"""
LOOP_RECOVERY_PROMPT_BANG = """
Loop recovery

Function Overview:
This prompt is designed to convert MLU multi-core indexing logic (with `clusterId` and `coreId`) into standard C++ `for` loop structures. 
The goal is to simulate the multi-core execution on a CPU using nested `for` loops, while preserving the logic for executing across different clusters and cores.

Application Scenario:
- Use this prompt when you need to convert MLU code, where the execution logic depends on multi-core and multi-cluster indexing, into sequential C++ code. This can be useful for debugging, CPU-based simulations of the original multi-core behavior, or porting the logic to platforms without multi-core NPUs.

### Input:
A function written for an MLU that uses `clusterId` and `coreId` for multi-core execution, and that requires converting this parallel structure into standard C++ `for` loops.

### Output:
The same logic rewritten using nested `for` loops to emulate the multi-core behavior of the MLU in a sequential manner.

### Steps for Conversion:
1. Identify the use of `clusterId` and `coreId` in the input code.
2. Convert the parallel multi-core structure into nested `for` loops in standard C++.
3. The number of clusters and cores are both 4.
4. Replace the multi-core indexing expressions with loop index variables (e.g., `clusterId * 4 + coreId` becomes a C++ loop with the same arithmetic).

### GPT Task:
Transform the following NPU code into equivalent C++ for-loop code that sequentially emulates the multi-cluster and multi-core indexing logic. The output should use nested `for` loops to replace the MLU indexing logic.
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
