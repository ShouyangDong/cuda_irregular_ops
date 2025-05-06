SPLIT_PRAGMA_PROMPT = """
You are an expert performance engineer with deep experience in optimizing numerical linear algebra kernels for high-performance computing systems.
Your primary goal is to optimize tensor operations to achieve peak efficiency on Deep Learning Processors (DLPs).
You leverage techniques such as loop unrolling, SIMD vectorization, memory locality optimization, cache management,
and hardware-specific parallelization strategies like tiling and pipelining.

Please analyze the following loop nest and determine which axes can be split using loop splitting.
Additionally, provide the appropriate pragma directive for loop splitting immediately above the for loop.

### Transformation Steps:
1. **Identify Axes for Loop Split**
   Locate the `for` loop(s) whose iteration space can be divided into equal chunks by a compile-time constant factor.
2. **Insert Pragma**
   Add a directive in the form `#pragma loop_split(factor)` immediately before the identified loop.

### Requirements:
- Do **not** modify any loop bodies or logic—only add the pragma.
- Use the syntax `#pragma loop_split(factor)`.
- If there are nested loops, choose the innermost or outermost loop as appropriate and explain.
- Do not change the computation logic.

### Input Code:
{code}

### Output Code:
Return the full C++ code, with the appropriate #pragma loop_split(...) inserted just before the for loop shown above.

"""

SPLIT_PRAGMA_DEMO = """

### Example 1:
Input code:
```cpp
void mul(float* A, float* B, float* C) {
    for (int i = 0; i < 60; i++) {
        A[i] = B[i] * C[i];
    }
}
```

Output code:
```cpp
void mul(float* A, float* B, float* C) {
    #pragma loop_split(factor)
    for (int i = 0; i < 60; i++) {
        A[i] = B[i] * C[i];
    }
}
```


### Example 2:
Input code:
```cpp
void add(float* A, float* B, float* C, int N, int M) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            C[i * M + j] = A[i * M + j] + B[i * M + j];
        }
    }
}
```

Output code:
```cpp
void add(float* A, float* B, float* C, int N, int M) {
    for (int i = 0; i < N; i++) {
        #pragma loop_split(factor)
        for (int j = 0; j < M; j++) {
            C[i * M + j] = A[i * M + j] + B[i * M + j];
        }
    }
}
```
### Example 3:
Input code:
```
extern "C" __mlu_global__ void add(float *lhs, float *rhs, float *out) {
    __nram__ float buf[512];
    // ...data copy...
    for (int k = 0; k < 512; ++k) {
        buf[k] = lhs[k] + rhs[k];
    }
    // ...write back...
}
```
Output code:
```
extern "C" __mlu_global__ void add(float *lhs, float *rhs, float *out) {
    __nram__ float buf[512];
    // ...data copy...
    #pragma loop_split(factor)
    for (int k = 0; k < 512; ++k) {
        buf[k] = lhs[k] + rhs[k];
    }
    // ...write back...
}
```
"""
