CACHE_READ_PROMPT = """
You are tasked with performing an optimization on C code that mimics the effect of `cache_read` from TVM in C for loops.
The goal is to buffer {buffer} into faster {CACHE_NAME} memory and adjust the loop accesses accordingly.


### Steps for conversion:
1. Identify repeated memory reads of {buffer} within the C code's for loops.
2. **Cache these reads** {buffer} into a {CACHE_NAME} buffer.
3. **Adjust the loop** to read from the cached buffer instead of the original memory location.
4. **Preserve any existing cache reads** from other buffers (e.g., if `A` is already cached, do not overwrite it).
If both buffers (e.g., `A` and `B`) need to be cached, **perform cache reads for both** buffers into separate buffers.
5. Output the transformed C code.

### Input:
The input will be C for loop code that accesses buffer {buffer} repeatedly. The loop bounds should be retained as per the input code.
The transformation must not assume a fixed loop boundary, but instead, retain the original boundary in the output.
Additionally, existing cache reads should not be removed.

{CODE}

### Requirements:
- The transformation should **maintain the original loop bounds** from the input code.
- Existing cached buffers (e.g., if `A` is cached in Nram) should be retained.
- If new buffers need to be cached (e.g., `B`), cache them in separate buffers.
- The transformation should mimic the behavior of `cache_read` from TVM and not alter the computation.
"""

CACHE_READ_DEMO = """
### Example Input:
```c
for (int i = 0; i < N; i++) {
    #pragma intrinsic(__bang_add(input[Nram, Nram], output[Nram]))
    for (int j = 0; j < M; j++) {
        C[i][j] = A[i][j] + B[i][j];
    }
}
```

### Example Output:
```c
for (int i = 0; i < N; i++) {
    {NAMESPACE} float A_{CACHE_NAME}[M];
    // Cache read of A into {CACHE_NAME} memory
    for (int j = 0; j < M; j++) {
        A_{CACHE_NAME}[j] = A[i][j];
    }

    #pragma intrinsic(__bang_add(input[Nram, Nram], output[Nram]))
    for (int j = 0; j < M; j++) {
        C[i][j] = A_{CACHE_NAME}[j] + B[i][j]; // Use cached version of A
    }
}
```

### Example Input:
```c
#pragma intrinsic(__bang_add(input[Nram, Nram], output[Nram]))
for (int j = 0; j < 64; j++) {
    C[j] = A[j] + B[j];
}

```

### Example Output:
```c
{NAMESPACE} float A_{CACHE_NAME}[64];
// Cache read of A into {CACHE_NAME} memory
for (int j = 0; j < 64; j++) {
    A_{CACHE_NAME}[j] = A[j];
}

#pragma intrinsic(__bang_add(input[Nram, Nram], output[Nram]))
for (int j = 0; j < 64; j++) {
    C[j] = A_{CACHE_NAME}[j] + B[j]; // Use cached version of A
}

```
"""

CACHE_WRITE_PROMPT = """
Cache Write：
You are tasked with performing an optimization on C code that mimics the effect of `cache_write` from TVM in C for loops.
The goal is to buffer intermediate results in {CACHE_NAME} memory (e.g., a temporary buffer)
during computation and write them back to the main memory once the computation is complete.

### Task:
1. **Identify the memory writes** in the innermost loop.
2. **Cache the write operations** into a {CACHE_NAME} buffer instead of writing directly to the original buffer.
3. **Write back the cached results** to the original buffer after the computation.
4. **Preserve any existing cache writes or reads** from other buffers (e.g., if `A` is already cached, do not overwrite it).
If both buffers (e.g., `A` and `B`) need to be cached, **perform cache reads for both** buffers into separate buffers.
5. Output the transformed C code.

### Input:
The input will be C for loop code that accesses buffers repeatedly. The loop bounds should be retained as per the input code.
The transformation must not assume a fixed loop boundary, but instead, retain the original boundary in the output.
Additionally, existing cache reads should not be removed.

{CODE}


### Requirements:
- The transformation should **maintain the original loop bounds** from the input code.
- Existing cached buffers (e.g., if `A` is cached in Nram) should be retained.
- If new buffers need to be cached (e.g., `B`), cache them in separate buffers.
- The transformation should mimic the behavior of `cache_read` from TVM and not alter the computation.
"""

CACHE_WRITE_DEMO = """
### Example Input:
```c
for (int i = 0; i < N; i++) {
    #pragma intrinsic(__bang_add(input[Nram, Nram], output[Nram]))
    for (int j = 0; j < M; j++) {
        C[i][j] = A[i][j] + B[i][j];
    }
}
```

### Example Output:
```c


for (int i = 0; i < N; i++) {
    {NAMESPACE} float C_{CACHE_NAME}[M];
    #pragma intrinsic(__bang_add(input[Nram, Nram], output[Nram]))
    for (int j = 0; j < M; j++) {
        C_{CACHE_NAME}[j] = A[i][j] + B[i][j]; // Store result in {CACHE_NAME} cache
    }
    // Write back the cached results to C
    for (int j = 0; j < M; j++) {
        C[i][j] = C_{CACHE_NAME}[j]; // Write cached result to original buffer
    }
}
```

### Example Input:
```c
#pragma intrinsic(__bang_add(input[Nram, Nram], output[Nram]))
for (int j = 0; j < 64; j++) {
    C[j] = A[j] + B[j];
}
```

### Example Output:
```c
{NAMESPACE} float C_{CACHE_NAME}[64];
#pragma intrinsic(__bang_add(input[Nram, Nram], output[Nram]))
for (int j = 0; j < 64; j++) {
    C_{CACHE_NAME}[j] = A[j] + B[j]; // Store result in {CACHE_NAME} cache
}
// Write back the cached results to C
for (int j = 0; j < 64; j++) {
    C[j] = C_{CACHE_NAME}[j]; // Write cached result to original buffer
}
```
"""

TENSORIZATION_PROMPT = """
Tensorization

Function Overview:
`TENSORIZATION` in the context of SIMD (Single Instruction, Multiple Data) is a technique that transforms scalar operations into vectorized operations to take advantage of the parallel processing capabilities of modern processors. By converting scalar computations (processing one element at a time) into tensorized or vectorized computations, SIMD instructions can process multiple data points simultaneously, improving throughput and reducing the overall computation time.

Application Scenario:
- Tensorization is widely used in deep learning frameworks to speed up matrix multiplications, convolutions, and other tensor operations by leveraging SIMD. For example, it can be used to vectorize the processing of large batches of input data, improving performance on CPUs, GPUs, and other accelerators.

- SIMD-based tensorization can be applied to common linear algebra kernels such as matrix-vector multiplications (GEMV), matrix-matrix multiplications (GEMM), and vector dot products. SIMD instructions accelerate these operations by processing multiple elements of vectors or matrices in parallel.
"""

TENSORIZATION_DEMO = """
Usage Examples:
// before:
#pragma operation(memory(input[output_nram], output[output]))
for (int i = 0; i < 512; ++i) {
    for (int j = 0; j < 512; ++j) {
        output[i * 512 + j] = output_nram[i * 512 + j];
    }
}

// after:
__memcpy(output, output_nram, 512 * 512 * 4, NRAM2GDRAM);
"""

DOUBLE_BUFFER_PROMPT = """
double buffering

Function Overview:
`DOUBLE_BUFFER` is a memory management and parallel processing technique designed to hide memory access latency by overlapping computation with data transfers. It utilizes two buffers, where one buffer is being read from or written to by the compute unit, while the other buffer is being populated with the next data to process. This ensures that the compute unit is never idle, leading to higher throughput and more efficient usage of hardware resources.

Application Scenario:
- In deep learning processors and GPUs, where large datasets need to be streamed to and from the compute units,
`DOUBLE_BUFFER` ensures that data movement does not stall computation.
For example, while one batch of input data is being processed, the next batch can be loaded into memory.
"""

DOUBLE_BUFFER_DEMO = """
Usage Examples:
input
```
__mlu_entry__ void add(float* INPUT0, float* INPUT1, float* OUTPUT) {
    __nram__ float INPUT0_N[64];
    __nram__ float INPUT1_N[64];
    __nram__ float OUTPUT_N[64];
    #pragma software_pipeline
    for (int i = 0; i < 2048; ++i) {
        __memcpy(INPUT0_N, INPUT0 + (i * 64), 256, GDRAM2NRAM);
        __memcpy(INPUT1_N, INPUT1 + (i * 64), 256, GDRAM2NRAM);
        __bang_add(OUTPUT_N, INPUT0_N , INPUT1_N, 64);
        __memcpy(OUTPUT + (i * 64), OUTPUT_N, 256, NRAM2GDRAM);
    }
}
```

output
```
__mlu_entry__ void fvec_add_double_bufferingfloat* INPUT0, float* INPUT1, float*OUTPUT) {
    __nram__ float INPUT0_N[128];
    __nram__ float INPUT1_N[128];
    __nram__ float OUTPUT_N[128];
    __memcpy_async(INPUT0_N, INPUT0, 256, GDRAM2NRAM);
    __asm__ volatile("sync;");
    __memcpy_async(INPUT1_N, INPUT1, 256, GDRAM2NRAM);
    __asm__ volatile("sync;");
    __memcpy_async(INPUT0_N + 64, INPUT0 + 64, 256, GDRAM2NRAM);
    __memcpy_async(INPUT1_N + 64, INPUT1 + 64, 256, GDRAM2NRAM);
    __bang_add(OUTPUT_N, INPUT0_N, INPUT1_N, 64);
    __asm__ volatile("sync;");
    for (int i_outer = 0; i_outer < 1023; ++i_outer) {
        __memcpy_async(INPUT0_N, INPUT0 + ((i_outer * 128) + 128), 256,GDRAM2NRAM);
        __memcpy_async(INPUT1_N, INPUT1 + ((i_outer * 128) + 128), 256,GDRAM2NRAM);
        __bang_add(OUTPUT_N + 64, INPUT0_N + 64, INPUT1_N + 64, 64);
        __memcpy_async(OUTPUT + (i_outer * 128), OUTPUT_N, 256,NRAM2GDRAM);
        __asm__ volatile("sync;");
        __memcpy_async(INPUT0_N + 64, INPUT0 + ((i_outer * 128) + 192),256, GDRAM2NRAM);
        __memcpy_async(INPUT1_N + 64, INPUT1 + ((i_outer * 128) + 192),256, GDRAM2NRAM);
        __bang_add(OUTPUT_N, INPUT0_N, INPUT1_N,64);
        __memcpy_async(OUTPUT + ((i_outer * 128) + 64), OUTPUT_N + 64, 256,NRAM2GDRAM);
        __asm__ volatile("sync;");
    }

    __bang_add(OUTPUT_N + 64, INPUT0_N + 64, INPUT1_N + 64,64);
    __memcpy_async(OUTPUT + 130944, OUTPUT_N, 256, NRAM2GDRAM);
    __asm__ volatile("sync;");
    __memcpy_async(OUTPUT + 131008, OUTPUT_N + 64, 256, NRAM2GDRAM);
    __asm__ volatile("sync;");}
```
"""

THREAD_BINDING_PROMPT_CUDA = """
Thread Binding

Function Overview:
You are tasked with identifying parallelizable loops in C++ code and binding them to the available block and thread on an GPU.
Maximum number of threads per block is 1024, and the maximum number of blocks per grid is 256.
Your goal is to transform the input code by mapping the loops onto specific hardware resources (block and thread) for parallel execution.

Application Scenario:
Use this prompt when you want to parallelize a computational task by binding one or more axes of a loop (e.g., batch size, spatial dimensions, etc.)
to the available threads in a GPU. This process accelerates the computation by exploiting the parallel nature of hardware accelerators.

Input:
The input is a C++/CUDA code snippet containing loops that can be parallelized, with the goal of binding these loops to threads on a GPU.
The target hardware may have specific threads, and the prompt will help map the loop dimensions accordingly.

Output:
The transformed code with appropriate thread binding directives inserted into the loops,
ensuring that each iteration of the loop is handled by different threads for parallel execution.


### Steps for Insertion:
1. **Identify parallelizable loops:** - Find the outermost or large loops suitable for parallelization. A loop is considered parallelizable if its iteration count is smaller than the number of blocks (256) or threads (1024).
2. Bind these loops to available hardware threads directly using CUDA constructs like `threadIdx.x` and `blockIdx.x`.
    For loops where the iteration count is larger than the number of block (256) or threads (1024),
    replace the loop variables with `blockIdx.x` and `threadIdx` respectively.
    - If the loop’s iteration count is smaller than 1024, add condition checks to ensure proper core binding.
    For example, `if (threadIdx.x < dim)`.
3. **Remove unnecessary loops:** - After replacing loop variables, remove the corresponding `for` loops for `threadIdx.x` and `blockIdx.x`, and directly map iterations to hardware blocks and threads.

### Example
{THREAD_BINDING_DEMO}

### GPT Task:
Please transform the following C++ by binding the parallel loops to GPU threads and cores for efficient parallel computation.

#### Input Code:
{cpp_code}

#### Output Code with Thread Binding:
```

### Notes:
- Replace the loop dimensions with `blockIdx.x` and `threadIdx` where possible.
- Ensure that the output code maintains the same computational logic while taking advantage of the parallel nature of the hardware.
- The variables `blockIdx.x` and `threadIdx` are built-in parallel variables and do not require initialization.
"""

THREAD_BINDING_PROMPT_BANG = """
Thread Binding

Function Overview:
You are tasked with identifying parallelizable loops in C++ code and binding them to the available clusters and cores on an NPU.
The target NPU has 4 clusters and 4 cores per cluster. Your goal is to transform the input code by mapping the loops onto specific
hardware resources (clusters and cores) for parallel execution.


Application Scenario:
Use this prompt when you want to parallelize a computational task by binding one or more axes of a loop to the available cores in a NPU.
This process accelerates the computation by exploiting the parallel nature of hardware accelerators.

Input:
The input is a C++/CUDA code snippet containing loops that can be parallelized. The goal is to bind these loops to cores and clusters on an NPU for parallel execution.
The target hardware has 4 clusters and 4 cores per cluster, and the prompt will help map the loop dimensions accordingly.

Output:
The transformed code should include appropriate thread binding directives, ensuring that iterations of the loop are distributed across the clusters and cores of the NPU for parallel execution.


 ### Steps for the transformation:
 1. **Identify parallelizable loops:** - Find the outermost or large loops suitable for parallelization. A loop is considered parallelizable if its iteration count is larger than the number of clusters (4) or cores (4).

 2. **Replace loop variables with `clusterId` and `coreId`:** - For loops where the iteration count is larger than the number of clusters (4) or cores (4), replace the loop variables with `clusterId` and `coreId` respectively. - If the loop’s iteration count is smaller than 4, add condition checks to ensure proper core binding. For example, `if (clusterId < dim)`.

 3. **Remove unnecessary loops:** - After replacing loop variables, remove the corresponding `for` loops for `clusterId` and `coreId`, and directly map iterations to hardware cores and clusters.


### Example
{THREAD_BINDING_DEMO}

### GPT Task:
Please transform the following C++ by binding the parallel loops to NPU clusters and cores for efficient parallel computation.

#### Input Code:
{cpp_code}

#### Output Code with Cluster Binding:
```

### Notes:
- Replace the loop dimensions with `clusterId` and `coreId` where possible.
- Ensure that the output code maintains the same computational logic while taking advantage of the parallel nature of the hardware.
- The variables `clusterId` and `coreId` are built-in parallel variables and do not require initialization.
"""


THREAD_BINDING_DEMO_BANG = """
### Example

### Input Code:
```cpp
#pragma thread_binding
for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4 j++) {
        for (int k = 0; k <  7; k++) {
            B[i * 4 * 7 + j * 7 + k] = A[i * 4 * 7 + j * 7 + k] + 1.0;
        }
    }
}
```

### Output Code with Cluster and Core Binding:
- Bind loops to **4 clusters** and **4 cores** on the NPU.
- Ensure parallel execution across clusters and cores.

Expected Output:

```cpp
for (int k = 0; k <  7; k++) {
    B[clusterId * 4 * 7 + coreId * 7 + k] = A[clusterId * 4 * 7 + coreId * 7 + k] + 1.0;
}
```
"""

THREAD_BINDING_DEMO_CUDA = """
Usage Examples:

Input CUDA/NPU C++ Code:
```cpp
for (int i = 0; i < 32; i++) {
    for (int j = 0; j < 1024; j++) {
        C[i][j] = A[i][j] + B[i][j];
    }
}
```

Desired Output Code with Thread/Cluster Binding:
```cpp
if (blockIdx.x < 32) {
    if (threadIdx.x < 1024) {
        C[blockIdx.x][threadIdx.x] = A[blockIdx.x][threadIdx.x] + B[blockIdx.x][threadIdx.x];
    }
}
```
"""

DECORATION_PROMPT = """
Operation Recognition

Function Overview:
Operation Recognition is designed to identify element-wise or matrix multiplication arithmetic
operations inside for loops in C++ code and insert the corresponding `#pragma operation( )` directives.
The inserted pragmas are intended to mark operations for future SIMD (Single Instruction, Multiple Data) vectorization.
This ensures that element-wise or matrix multiplication calculations can be efficiently transformed into SIMD instructions
during a later code transformation stage.

### Application Scenario:
Use this prompt when preparing C++ code for SIMD tensorization. It helps identify and mark arithmetic operations inside for loops that operate on individual elements of buffers or matrices. These operations will be optimized and vectorized in the later stages.

### Input:
The input is a C++ code snippet containing for loops with element-wise or matrix multiplication arithmetic operations, where you want to insert `#pragma operation( )` directives before each operation for SIMD vectorization purposes.

### Output:
The transformed C++ code with the `#pragma operation( )` directives inserted before the detected operations and arguments inside loops, which marks them for SIMD vectorization.

### Example:

#### Input C++ Code:
```cpp
for (int i = 0; i < 64; i++) {
    C[i] = A[i] + B[i];
}

for (int i = 0; i < 64; i++) {
    C[i] = C[i] * D[i];
}

for (int i = 0; i < 64; i++) {
    for (int j = 0; j < 128; j++) {
        E[i*128+j] = C[i*128+j] - D[i*128+j];
    }
}

for (int i = 0; i < 512; i++) {
    A_nram[i] = A[(clusterId * 4 + coreId) * 512 + i];
}

for (int col = 0; col < 64; col++) {
    C[(clusterId * 4 + coreId) * 64 + col] = C_wram[col];
}
```

#### Desired Output C++ Code with Pragmas for SIMD Preparation:
```cpp
#pragma operation(add(input[A, B], output[C]))
for (int i = 0; i < 64; i++) {
    C[i] = A[i] + B[i];
}
#pragma operation(mul(input[C, D], output[C]))
for (int i = 0; i < 64; i++) {
    C[i] = C[i] * D[i];
}

for (int i = 0; i < 64; i++) {
    #pragma operation(sub(input[C, D], output[E]))
    for (int j = 0; j < 128; j++) {
        E[i*128+j] = C[i*128+j] - D[i*128+j];
    }
}

#pragma operation(memory(input[A], output[A_nram]))
for (int i = 0; i < 512; i++) {
    A_nram[i] = A[(clusterId * 4 + coreId) * 512 + i];
}

#pragma operation(memory(input[C_nram], output[C]))
for (int col = 0; col < 64; col++) {
    C[(clusterId * 4 + coreId) * 64 + col] = C_wram[col];
}
```

### Steps for Insertion:
1. Identify element-wise or matrix multiplication arithmetic operations inside the for loop such as addition (`+`), subtraction (`-`), multiplication (`*`), and division (`/`).
2. Insert the corresponding `#pragma operation( )` directive directly above each identified operation, specifying the operation type in parentheses (e.g., `#pragma operation(add)` for addition).
3. Focus only on the operations inside loops, as these are the target for SIMD tensorization.
4. Ensure that the structure and logic of the code are not altered, and only relevant element-wise or matrix multiplication operations are annotated.

### GPT Task:
Please transform the following C++ code by inserting `#pragma operation( )` directives above each element-wise or matrix multiplication arithmetic operation inside for loops. These pragmas will be used to prepare the code for SIMD vectorization in a later stage.

#### Input C++ Code:
{cpp_code}

#### Output C++ Code with Pragmas for SIMD Preparation:
```

### Notes:
- The input should be replaced with the actual input C++ code containing loops with element-wise or matrix multiplication operations.
- The output should focus on identifying and marking operations inside loops that are candidates for SIMD vectorization.
- The Pragmas must be above the for loops.
"""
