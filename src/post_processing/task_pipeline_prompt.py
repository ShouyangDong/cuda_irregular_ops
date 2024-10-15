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
---

### **Usage Examples**:

#### **Example 1: CACHE_READ in CUDA Programming**
```cpp
__global__ void matrix_mul(float* A, float* B, float* C, int N) {
    __shared__ float shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_B[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    float sum = 0.0f;
    for (int i = 0; i < N / TILE_SIZE; i++) {
        // Cache the matrix data in shared memory for faster read access
        shared_A[ty][tx] = A[row * N + i * TILE_SIZE + tx];
        shared_B[ty][tx] = B[(i * TILE_SIZE + ty) * N + col];
        __syncthreads();
        
        for (int j = 0; j < TILE_SIZE; j++) {
            sum += shared_A[ty][j] * shared_B[j][tx];
        }
        __syncthreads();
    }

    C[row * N + col] = sum;
}
```
"""
CACHE_WRITE_PROMPT = """
"""
CACHE_WRITE_DEMO = """
"""
TENSORIZATION_PROMPT = """
"""
TENSORIZATION_DEMO = """
"""
DETENSORIZATION_PROMPT = """
"""
DETENSORIZATION_DEMO = """
"""

DOUBLE_BUFFER_PROMPT = """
"""
DOUBLE_BUFFER_DEMO = """
"""
