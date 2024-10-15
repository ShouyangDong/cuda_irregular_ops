LOOP_FUSION_PROMPT = """
Loop fusion

Function Overview: 
Fuse multiple independent or partially dependent `for` loops into a single loop to minimize 
loop overhead and improve cache efficiency. You should ensure that the transformation is correct, 
preserves program semantics, and handles any dependencies or constraints between the loops. 


Application Scenario: 
- When multiple loops iterate over the same or related data structures, loop fusion combines them into a single loop.
- Loop fusion helps to streamline computation and reduce time spent on memory access.
"""

LOOP_FUSION_DEMO = """
Usage Examples:

Input:
```cpp
for (int i = 0; i < 300; i++) {
    for (int j = 0; j < 300; j++)
        a[i * 300 + j] = b[i * 300 + j] + 4;
    }
}

Output:
```cpp
for (int i_fuse = 0; i_fuse < 300 * 300; i_fuse++) {
    a[i_fuse] = b[i_fuse] + 4;
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

Input:
```cpp
for (int i = 0; i < N; i++) { 
    for (int j = 0; j < N; j++) { 
        A[i][j] = B[i][j] + C[i][j]; 
    }
}

Output:
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

Original loop:
```cpp
for (int i = 0; i < N; i++) {
    A[i] = B[i] * C[i];
}
```

output:
```cpp
int tile_size = 64;
for (int t = 0; t < N; t += tile_size) {
    for (int j = t; j < std::min(t + tile_size, N); j++) {
        A[i] = B[i] * C[i];
    }
}
```
"""

THREAD_BINDING_PROMPT = """
Thread binding

Function Overview:
`THREAD_BINDING` is a parallel computing optimization technique used to control the mapping of 
threads to specific hardware resources, such as CPU cores or GPU execution units. 
By binding or pinning threads to specific hardware threads or cores, 
the technique leads to better cache locality, memory access patterns, and overall system performance.


Application Scenario:
- In scenarios where different cores are handling different tasks, binding threads to specific cores 
  can minimize cache misses and maximize the usage of per-core resources. 
  This is important in high-performance applications where CPU utilization 
  needs to be balanced and consistent.
  
- Thread binding is crucial in GPU programming to control how threads are distributed across the 
  various compute units (CUs) or streaming multiprocessors (SMs). 
  This allows optimal utilization of shared memory, registers, and caches in GPU architectures,
  ensuring efficient execution of parallel workloads.
"""

THREAD_BINDING_DEMO = """
```cpp
__kernel void matrix_mul(__global float* A, __global float* B, __global float* C, int N) {
    int row = get_global_id(0); // Global thread index
    int col = get_global_id(1);

    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += A[row * N + i] * B[i * N + col];
    }
    C[row * N + col] = sum;
}
}
```
"""
"TENSOR_COMTRACTION",
