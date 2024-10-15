DETENSORIZATION_PROMPT = """
"""
DETENSORIZATION_DEMO = """
exmaple 1:
```
__memcpy((float *)input0_local_nram + (0), (float *)input0 + (((clusterId * 256) + (coreId * 64))), 64, GDRAM2NRAM);
```
converted to:
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
input:
```
extern "C" __global__ void __launch_bounds__(640) sign_kernel(float* __restrict__ A, float* __restrict__ T_sign) {
    T_sign[((int)threadIdx.x)] = ((0.000000e+00f < A[((int)threadIdx.x)]) ? 1.000000e+00f : ((A[((int)threadIdx.x)] < 0.000000e+00f) ? -1.000000e+00f : 0.000000e+00f));
}
```
output:
```
extern "C" void sign_kernel(float* A, float* T_sign) {
for (int threadIdx.x = 0; threadIdx.x < 640; threadIdx.x++) {
    T_sign[threadIdx.x] = ((0.000000e+00f < A[threadIdx.x]) ? 1.000000e+00f : ((A[threadIdx.x] < 0.000000e+00f) ? -1.000000e+00f : 0.000000e+00f));
    }
}
```
Example 2:
input:
```
extern "C" __global__ void __launch_bounds__(1024) add_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ T_add) {
    if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 2309) {
        T_add[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (A[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] + B[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
    }
}
```
output:
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
input:
```
for (int i = 0; i < 64; i++) {
    output[coreId * 64 + i] += A[coreId * 64 + i] * 2;
}
```
output:
```
for (int coreId = 0; coreId < 4; coreId++) {
    for (int i = 0; i < 64; i++) {
        output[coreId * 64 + i] += A[coreId * 64 + i] * 2;
    }
}
```
"""
