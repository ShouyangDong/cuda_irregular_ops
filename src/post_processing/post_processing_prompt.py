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
input: 
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
output:

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

"""
CACHE_WRITE_DEMO = """
Usage Examples:
input:
extern "C" void abs_kernel(float *A, float *B) {
    for (int i = 0; i < 512; ++i) {
        for (int j = 0; j < 512; ++j) {
            B[i * 512 + j] = A[i * 512 + j] + 1.0;
        }
    }
}

buffer region: B
scope: NRAM

output:
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
"""
TENSORIZATION_DEMO = """
Usage Examples:
input:
for (int i = 0; i < 512; ++i) {
    for (int j = 0; j < 512; ++j) {
        output[i * 512 + j] = output_nram[i * 512 + j];
    }
}

output: 
__memcpy(output, output_nram, 512 * 512 * 4, NRAM2GDRAM);
"""


DOUBLE_BUFFER_PROMPT = """
"""
DOUBLE_BUFFER_DEMO = """
"""
