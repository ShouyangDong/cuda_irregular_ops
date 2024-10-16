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
"""

LOOP_RECOVERY_DEMO_CUDA = """
Example 1:
// before:
```
extern "C" __global__ void __launch_bounds__(640) sign_kernel(float* __restrict__ A, float* __restrict__ T_sign) {
    T_sign[((int)threadIdx.x)] = ((0.000000e+00f < A[((int)threadIdx.x)]) ? 1.000000e+00f : ((A[((int)threadIdx.x)] < 0.000000e+00f) ? -1.000000e+00f : 0.000000e+00f));
}
```
// after:
```
extern "C" void sign_kernel(float* A, float* T_sign) {
for (int threadIdx.x = 0; threadIdx.x < 640; threadIdx.x++) {
    T_sign[threadIdx.x] = ((0.000000e+00f < A[threadIdx.x]) ? 1.000000e+00f : ((A[threadIdx.x] < 0.000000e+00f) ? -1.000000e+00f : 0.000000e+00f));
    }
}
```
Example 2:
// before:
```
extern "C" __global__ void __launch_bounds__(1024) add_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ T_add) {
    if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 2309) {
        T_add[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (A[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] + B[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
    }
}
```
// after:
```
extern "C" void add_kernel(float* A, float* B, float* T_add) {
    for (int blockIdx.x = 0; blockIdx.x < 32; blockIdx.x++) {
        for (int threadIdx.x = 0; threadIdx.x < 1024; threadIdx.x++) {
            if (((blockIdx.x * 4) + (threadIdx.x >> 8)) < 9) {
                T_add[((blockIdx.x * 1024) + threadIdx.x)] = (A[((blockIdx.x * 1024) + threadIdx.x)] + B[((blockIdx.x * 1024) + threadIdx.x)]);
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
