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
- For matrix multiplications, convolutions, or stencil computations, loop reordering can increase data locality, 
reducing memory latency. 
- Reordering loops in scientific computing kernels can optimize cache usage and improve performance. 
"""

LOOP_REORDER_DEMO = """
Usage Examples:

// before:
```cpp
#pragma loop_reorder
for (int i = 0; i < N; i++) { 
    for (int j = 0; j < N; j++) { 
        A[i][j] = B[i][j] + C[i][j]; 
    }
}

// after:
```cpp
for (int j = 0; j < N; j++) { 
    for (int i = 0; i < N; i++) { 
        A[i][j] = B[i][j] + C[i][j]; 
    }
} 
"""

LOOP_SPLIT_PROMPT = """
Loop split

Function Overview:
`LOOP_SPLIT` is a loop transformation technique used to break a single loop into two or more separate loops. 
This technique can enhance performance by enabling better control over memory access, parallelization, 
and instruction-level optimizations. 


Application Scenario:
- In memory-bound applications, splitting a loop can improve data locality, reducing cache misses by working on smaller chunks of data.
- For multicore systems or GPUs, splitting loops can facilitate parallel execution by distributing the work across multiple processing units.
- Loops with conditionals can be split so that different conditions are handled in separate loops, improving the predictability of the loop behavior for the compiler.
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
for (i.outer, 0, 4) {
    for (i.inner, 0, 16) {
        if ((i.inner + i.outer*16) < 60) {
            A[i.inner + i.outer*16] = B[i.inner + i.outer*16] + B[i.inner + i.outer*16]
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
"""

TENSOR_COMTRACTION_DEMO = """
Usage Examples:
before:
```cpp
#pragma tensor_contraction
if (((clusterId * 4) + coreId) < 5) {
    for (int i = 0; i < 672; i++) {
        if (input0_local_nram[i] >= 0.0f) {
            input0_local_nram[i] = 1.0f;
        } else {
            input0_local_nram[i] = -1.0f;
        }
    }
}
if (((clusterId * 4) + coreId) < 5) {
    for (int i = 0; i < 2688/sizeof(float); i++) {
        active_sign_268[(((clusterId * 2688) + (coreId * 672))) + i] = input0_local_nram[i];
    }
}
```
after
```cpp
if (((clusterId * 4) + coreId) < 5) {
    for (int i = 0; i < 672; i++) {
        if (input0_local_nram[i] >= 0.0f) {
            active_sign_268[(((clusterId * 2688) + (coreId * 672))) + i] = 1.0f;
        } else {
            active_sign_268[(((clusterId * 2688) + (coreId * 672))) + i] = -1.0f;
        }
    }
}
```
"""
