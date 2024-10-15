CACHE_READ_PROMPT = """
cache read

Function Overview:
`CACHE_READ` is a memory optimization technique used to control how data is read 
from cache memory in systems that use hierarchical memory structures (e.g., CPUs, GPUs). 
It ensures that data is loaded efficiently from different levels of cache or main memory 
into registers or local memory to minimize memory latency and maximize performance.

Application Scenario:
- In GPU workloads or NPU workloads, where threads operate in parallel and data needs to be 
reused frequently within a thread block, `CACHE_READ` optimizes the data fetching from global 
memory to shared memory or registers.
"""

CACHE_READ_DEMO = """
Usage Examples:

```cpp
// before: 
```
extern "C" void abs_kernel(float *A, float *B, float *C) {
    for (int i = 0; i < 512; ++i) {
        for (int j = 0; j < 512; ++j) {
            C[i * 512 + j] = A[i * 512 + j] + B[i * 512 + j];
        }
    }
}
```
buffer region: A
scope: NRAM
// after:

```
extern "C" void abs_kernel(float *A, float *B, float *C) {
    __nram__ float A[512 * 512];
    for (int i = 0; i < 512; ++i) {
        for (int j = 0; j < 512; ++j) {
            A_nram[i * 512 + j] = A[i * 512 + j];
        }
    }

    for (int i = 0; i < 512; ++i) {
        for (int j = 0; j < 512; ++j) {
            C[i * 512 + j] = A_nram[i * 512 + j] + B[i * 512 + j];
        }
    }
}
```
"""

CACHE_WRITE_PROMPT = """
cache write

Function Overview:
`CACHE_WRITE` is an optimization technique that allows data to be written directly to a cache, 
ensuring faster access during future computations and reducing the overhead of writing to slower main memory. 
By buffering writes in the cache, it improves memory locality, increases the overall performance of the program, 
and minimizes memory bottlenecks. 

Application Scenario:
- Deep learning frameworks often perform multiple passes over large datasets. For operations such as backpropagation, `CACHE_WRITE` helps in caching gradients or intermediate results during training, thus speeding up the memory access and reducing the time spent writing to memory.
"""

CACHE_WRITE_DEMO = """
Usage Examples:
// before:
extern "C" void abs_kernel(float *A, float *B) {
    for (int i = 0; i < 512; ++i) {
        for (int j = 0; j < 512; ++j) {
            B[i * 512 + j] = A[i * 512 + j] + 1.0;
        }
    }
}

buffer region: B
scope: NRAM

// after:
extern "C" void abs_kernel(float *A, float *B) {
    __nram__ float B[512 * 512];
    for (int i = 0; i < 512; ++i) {
        for (int j = 0; j < 512; ++j) {
            B_nram[i * 512 + j] = A[i * 512 + j];
        }
    }

    for (int i = 0; i < 512; ++i) {
        for (int j = 0; j < 512; ++j) {
            B[i * 512 + j] = B_nram[i * 512 + j];
        }
    }
}
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
