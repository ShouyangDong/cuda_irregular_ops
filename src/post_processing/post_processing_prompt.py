CACHE_READ_PROMPT = """
You are tasked with performing an optimization on C code that mimics the effect of `cache_read` from TVM in C for loops. 
The goal is to cache data into faster {CACHE_NAME} memory and adjust the loop accesses accordingly. 


### Steps for conversion:
1. **Identify repeated memory reads** within the C code's for loops.
2. **Cache these reads** into a {CACHE_NAME} array (e.g., simulating shared memory or {CACHE_NAME} memory).
3. **Adjust the loop** to read from the cached array instead of the original memory location.
4. Output the transformed C code.

### Input:
The input will be C for loop code that accesses arrays repeatedly, as shown below:
```c
for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
        C[i][j] = A[i][j] + B[i][j];
    }
}
```

### Output:
Transform the code by introducing cache {CACHE_NAME} for one or more of the arrays, as in the following steps:
1. Load the array into a temporary cache before the innermost loop.
2. Access the cache instead of the original memory location in the inner loop.

### Requirements:
- The transformation must mimic the behavior of `cache_read` from TVM.
- All caching should be done before the innermost loop starts.
- The transformation should not alter the semantics of the computation.
"""

CACHE_READ_DEMO = """
### Example Input:
```c
for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
        C[i][j] = A[i][j] + B[i][j];
    }
}
```

### Example Output:
```c
for (int i = 0; i < N; i++) {
    // Cache read of A into {CACHE_NAME} memory
    {NAMESPACE} float A_{CACHE_NAME}[M];
    for (int j = 0; j < M; j++) {
        A_{CACHE_NAME}[j] = A[i][j];
    }

    for (int j = 0; j < M; j++) {
        C[i][j] = A_{CACHE_NAME}[j] + B[i][j]; // Use cached version of A
    }
}
```
"""

CACHE_WRITE_PROMPT = """
Cache Write： 
You are tasked with performing an optimization on C code that mimics the effect of `cache_write` from TVM in C for loops. 
The goal is to buffer intermediate results in {CACHE_NAME} memory (e.g., a temporary array) 
during computation and write them back to the main memory once the computation is complete.

### Task:
1. **Identify the memory writes** in the innermost loop.
2. **Cache the write operations** into a {CACHE_NAME} array  instead of writing directly to the original array.
3. **Write back the cached results** to the original array after the computation.

### Input:
The input will be C for loop code where arrays are written to inside the loop, as shown below:
```c
for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
        C[i][j] = A[i][j] + B[i][j];
    }
}
```

### Output:
Transform the code by introducing a cache {CACHE_NAME} to buffer the write operations, and then write the cached results back to the original memory location after the loop. 

### Requirements:
- The transformation should buffer the write operations to a temporary {CACHE_NAME} array.
- After the innermost loop finishes, the results in the {CACHE_NAME} cache should be written back to the original array.
- The transformation should not alter the semantics of the computation.
"""

CACHE_WRITE_DEMO = """
### Example Input:
```c
for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
        C[i][j] = A[i][j] + B[i][j];
    }
}
```

### Example Output:
```c
for (int i = 0; i < N; i++) {
    // Cache writes to C in {CACHE_NAME} memory
    {NAMESPACE} float C_{CACHE_NAME}[M];
    
    for (int j = 0; j < M; j++) {
        C_{CACHE_NAME}[j] = A[i][j] + B[i][j]; // Store result in {CACHE_NAME} cache
    }

    // Write back the cached results to C
    for (int j = 0; j < M; j++) {
        C[i][j] = C_{CACHE_NAME}[j]; // Write cached result to original array
    }
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
#pragma operation(memory store to output)
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
__mlu_entry__ void fvec_add_double_buffering_kernel0(float* INPUT0, float* INPUT1, float*OUTPUT) {
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

THREAD_BINDING_PROMPT = """
Thread Binding

Function Overview:
This prompt is designed to identify parallelizable loops or axes in C++ 
and bind them to the available threads or cores on a GPU or NPU. The prompt helps 
transform the input code by mapping the loops onto specific hardware resources like GPU threads or NPU cores 
to enable parallel computation.

Application Scenario:
Use this prompt when you want to parallelize a computational task by binding one or more axes of a loop (e.g., batch size, spatial dimensions, etc.) 
to the available threads or cores in a GPU/NPU. This process accelerates the computation by exploiting the parallel nature of hardware accelerators.

Input:
The input is a C++/CUDA code snippet containing loops that can be parallelized, with the goal of binding these loops to threads or cores on a GPU/NPU. 
The target hardware may have specific clusters, cores, or threads, and the prompt will help map the loop dimensions accordingly.

Output:
The transformed code with appropriate `#pragma` or thread binding directives inserted into the loops, 
ensuring that each iteration of the loop is handled by different threads or cores for parallel execution.


### Steps for Insertion:
1. Identify loops or axes that are candidates for parallel execution. Typically, outer loops or large iterations are ideal for parallelization.
2. Bind these loops to available hardware threads or cores using directives such as `#pragma thread_binding` or directly using CUDA constructs like `threadIdx` and `blockIdx`.
3. For NPU hardware, bind the loops to clusters and cores (e.g., clusterId, coreId).
4. Maintain the code logic, ensuring that the transformed code remains functionally equivalent while parallelizing the computation.

### Example 
{LOOP_RECOVERY_DEMO}

### GPT Task:
Please transform the following C++ or CUDA code by binding the parallel loops to GPU threads or NPU clusters and cores for efficient parallel computation. Insert `#pragma thread_binding` or equivalent GPU/NPU constructs where appropriate.

#### Input Code:
{cpp_code}

#### Output Code with Thread/Cluster Binding:
```

### Notes:
- `{cpp_code}` should be replaced with the actual input C++/CUDA code containing loops that are suitable for parallelization.
- The output should map parallel loops to the hardware resources available on the target device (e.g., GPU threads, NPU clusters/cores).
- The prompt is flexible enough to handle both GPU (CUDA) and NPU-specific architectures.

"""

THREAD_BINDING_DEMO_BANG = """
Usage Examples:

before
```cpp
#pragma thread_binding
for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4 j++) {
        B[i * 4 + j] = A[i * 4 + j] + 1.0;
    }
}
```

after
```cpp
B[clusterId * 4 + coreId] = A[clusterId * 4 + coreId] + 1.0;
```
"""

THREAD_BINDING_DEMO_CUDA = """
Usage Examples:

Input CUDA/NPU C++ Code:
```cpp
for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
        C[i][j] = A[i][j] + B[i][j];
    }
}
```

Desired Output Code with Thread/Cluster Binding:
```cpp
#pragma thread_binding(threadIdx.x, blockIdx.x)
for (int i = 0; i < N; i++) {
    #pragma thread_binding(threadIdx.y, blockIdx.y)
    for (int j = 0; j < M; j++) {
        C[i][j] = A[i][j] + B[i][j];
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
Use this prompt when preparing C++ code for SIMD tensorization. It helps identify and mark arithmetic operations inside for loops that operate on individual elements of arrays or matrices. These operations will be optimized and vectorized in the later stages.

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
    E[i] = C[i] - D[i];
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
#pragma operation(sub(input[C, D], output[E]))
for (int i = 0; i < 64; i++) {
    E[i] = C[i] - D[i];
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
"""
