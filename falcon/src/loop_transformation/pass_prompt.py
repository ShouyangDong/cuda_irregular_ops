LOOP_FUSION_PROMPT = """
Loop fusion


You are tasked with performing loop fusion on the provided C++ code. Loop fusion is a technique that combines multiple loops that iterate over the same range into a single loop to enhance performance by improving data locality and reducing overhead.

### Transformation Steps:
1. **Identify Candidate Loops**: Look for `for` loops that operate over the same index ranges and perform independent operations on arrays.

2. **Flatten the Loop Structure**: Replace the nested loops with a single loop that computes the combined index. This involves calculating a new index that represents the unique combination of the original loop indices.

3. **Adjust the Loop Body**: Ensure that the operations within the fused loop reflect the original operations from the individual loops.

4. **Remove the Original Loops**: After fusion, remove the original loops to avoid redundant computation.

### Notes:
- Ensure that the loops operate on the same data structure and index ranges.
- Make sure that the original logic and computations are preserved after fusion.
- Handle array indexing carefully to avoid out-of-bounds errors.

### Input Code:
Please provide the C++ code containing nested `for` loops suitable for loop fusion.

{code}

### Output Code:
Return the transformed C++ code after applying loop fusion.
"""

LOOP_FUSION_DEMO = """
Usage Examples:

// before:
```cpp
#pragma loop_fusion
for (int i = 0; i < 300; i++) {
    for (int j = 0; j < 300; j++)
        a[i * 300 + j] = b[i * 300 + j] + 4;
    }
}

// after:
```cpp
for (int i_j_fuse = 0; i_j_fuse < 300 * 300; i_j_fuse++) {
    a[i_j_fuse] = b[i_j_fuse] + 4;
}
"""

LOOP_REORDER_PROMPT = """
Loop reorder

Function Overview:
`LOOP_REORDER` is an optimization technique that reorders the sequence of loops in nested loops structures.
The primary goal of reordering loops is to enhance data locality and improve cache utilization,
leading to better performance on hardware such as CPUs, GPUs, or other accelerators.
By modifying the loop order, this technique can reduce cache misses and take advantage of memory hierarchies,
resulting in faster execution for data-intensive applications.

Application Scenario:
- In memory-bound applications, splitting a loop can improve data locality, reducing cache misses by working on smaller chunks of data.
- For multicore systems or GPUs, splitting loops can facilitate parallel execution by distributing the work across multiple processing units.

### Transformation Steps:
1. **Analyze Loop Dependencies**: Ensure that the inner loop (the loop over `j`) does not depend on the execution order of the outer loop (the loop over `i`).
2. **Check Array Access Patterns**: Make sure that reordering the loops does not affect data locality. For example, prioritize accessing adjacent memory locations to improve cache utilization.
3. **Adjust Loop Structure**: Swap the order of the inner and outer loops.
4. **Update Indices**: Ensure that all array accesses remain correct in the new loop structure.

### Considerations:
- **Data Dependency**: Ensure that there are no data dependencies between the reordered loops; otherwise, it could compromise the correctness of the program.
- **Cache Performance**: Reordering may affect cache locality. Try to maintain contiguous memory access to optimize performance.
- **Testing and Validation**: Conduct thorough testing after reordering to ensure that the output remains consistent with the original code.

### Input Code:
Please provide the C++ code containing nested `for` loops suitable for loop reorder.

{code}

### Output Code:
Return the transformed C++ code after applying loop reorder.
"""

LOOP_REORDER_DEMO = """
Example：

### Original Code:
```cpp
for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
        C[i * M + j] = A[i * M + j] + B[i * M + j];
    }
}
```

### Target Code:
```cpp
for (int j = 0; j < M; j++) {
    for (int i = 0; i < N; i++) {
        C[i * M + j] = A[i * M + j] + B[i * M + j];
    }
}
```
"""

LOOP_SPLIT_PROMPT = """
Please split the following loop based on the parameter specified in the `#pragma loop_split` directive.

### Original Code:
```cpp
#pragma loop_split(4)
for (int i = 0; i < 60; i++) {
    A[i] = B[i] * C[i];
}
```

### Transformation Steps:
1. **Identify the Split Factor**: The split factor is specified as `4`, indicating that the loop variable `i` should be split into 4 outer iterations.
2. **Calculate the Iteration Range**: Given that the original loop iterates from `0` to `59`, the outer loop should iterate 4 times, and the inner loop should cover the corresponding range of iterations. Each outer iteration will cover `60 / 4 = 15` iterations of the inner loop.

3. **Create the New Loop Structure**: The outer loop will iterate over `k` from `0` to `3`, and the inner loop will iterate over `j` from `0` to `14`.

### After Transformation:
```cpp
for (int k = 0; k < 4; k++) {
    for (int j = 0; j < 15; j++) {
        A[k * 15 + j] = B[k * 15 + j] * C[k * 15 + j];
    }
}
```

### Input Code:
Please provide the C++ code containing nested `for` loops suitable for loop split.

{code}

### Output Code:
Return the transformed C++ code after applying loop split.
"""

LOOP_SPLIT_DEMO = """
Usage Examples:

before:
```cpp
#pragma loop_split(factor=4)
for (int i = 0; i < 60; i++) {
    A[i] = B[i] * C[i];
}
```

// after:
```cpp
for (int k = 0; k < 4; k++) {
    for (int j = 0; j < 15; j++) {
        if ((k * 15 + j) < 60) {
            A[k * 15 + j] = B[k * 15 + j] * C[k * 15 + j];
        }
    }
}

```
"""


TENSOR_CONTRACTION = """
Tensor Contraction

Function Overview:
Tensor contraction is a technique used to merge two loops that share the same size and stride into a single loop.
By combining these loops, the overall computational workload can be reduced,
and memory access patterns can be optimized, leading to improved performance in tensor computations.

Application Scenario:
Tensor contraction is commonly applied in scenarios where two or more loops operate over dimensions with identical sizes and memory strides. This is particularly useful in optimizing tensor operations in deep learning models, such as matrix multiplications, convolutions, and backpropagation. By reducing the number of loops, tensor contraction helps minimize the overhead of loop management and improves the efficiency of data accesses, which is critical in handling large-scale tensor computations in fields like scientific computing, machine learning, and quantum physics.

Note:
Please apply tensor contraction to the following complete C/C++ kernel function (it may be annotated with `__mlu_global__`, `__global__`, or similar).
Return the **entire** transformed function—signature, qualifiers, all braces, and any comments—wrapped in a ```cpp code block. Do **not** output only a fragment.
"""

TENSOR_CONTRACTION_DEMO = """

Example 1: Fuse two simple element-wise loops

before:
```cpp
extern \"C\" void clamp_and_copy(float* data, float* out, int N) {
    for (int i = 0; i < N; ++i) {
        // clamp to [0,1]
        if (data[i] < 0.0f) data[i] = 0.0f;
        else if (data[i] > 1.0f) data[i] = 1.0f;
    }
    for (int i = 0; i < N; ++i) {
        // copy to output
        out[i] = data[i];
    }
}
```

after:
```cpp
extern \"C\" void clamp_and_copy(float* data, float* out, int N) {
    for (int i = 0; i < N; ++i) {
        // clamp to [0,1]
        if (data[i] < 0.0f) data[i] = 0.0f;
        else if (data[i] > 1.0f) data[i] = 1.0f;
        out[i] = data[i];
    }
}
```
---
Example 2: Merge two memcpy loops and a compute in host code

before:
```cpp
void transform_and_write(float* src, float* dst, int M) {
    float buf[128];
    for (int i = 0; i < M; ++i) {
        memcpy(buf, src + i*128, 128*sizeof(float));
    }
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < 128; ++j) {
            buf[j] = buf[j] * 2.0f;
        }
    }
    for (int i = 0; i < M; ++i) {
        memcpy(dst + i*128, buf, 128*sizeof(float));
    }
}
```

after:
```cpp
void transform_and_write(float* src, float* dst, int M) {
    float buf[128];
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < 128; ++j) {
            // load, scale, and store in one pass
            float v = src[i*128 + j];
            buf[j] = v * 2.0f;
            dst[i*128 + j] = buf[j];
        }
    }
}
```
---
**Example 3: Merge only for-loops, not if conditions**

**before:**
```cpp
extern "C" void relu_and_store(float* input, float* output, int N) {
    if (N > 0) {
        for (int i = 0; i < N; ++i) {
            input[i] = input[i] > 0 ? input[i] : 0;
        }
    }
    if (N > 0) {
        for (int i = 0; i < N; ++i) {
            output[i] = input[i];
        }
    }
}
```

**after (loops merged, ifs preserved):**
```cpp
extern "C" void relu_and_store(float* input, float* output, int N) {
    if (N > 0) {
        for (int i = 0; i < N; ++i) {
            input[i] = input[i] > 0 ? input[i] : 0;
            output[i] = input[i];
        }
    }
}
```
"""
