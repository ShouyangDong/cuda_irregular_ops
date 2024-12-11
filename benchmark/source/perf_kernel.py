import torch
import glob
import os
import torch.nn.functional as F
torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.deterministic = True
device = torch.device("cuda")
# 创建随机张量，数据类型为float16
# shapes = [
# [128, 128, 128],
# [128, 256, 512],
# [256, 256, 256],
# [512, 512, 512],
# [128, 512, 256],
# [256, 512, 256],
# [512, 512, 256],
# [256, 128, 256],
# ]
times = []
files = glob.glob(
    os.path.join(os.getcwd(), "benchmark/data/dlboost_code_test/mha*.cpp")
)
for file in files:
    base_name = os.path.basename(file)
    shapes = base_name.split(".")[0]
    shape = [int(intg) for intg in shapes.split("_")[1:]]
    # Optionally use the context manager to ensure one of the fused kernels is run
    query = torch.rand(shape, dtype=torch.float16, device=device)
    key = torch.rand(shape, dtype=torch.float16, device=device)
    value = torch.rand(shape, dtype=torch.float16, device=device)

    def test_scaled_dot_product_attention():
        output = F.scaled_dot_product_attention(query, key, value)
        

    for _ in range(100):
        test_scaled_dot_product_attention()

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    with torch.profiler.profile(activities=(torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA),
                record_shapes=True, on_trace_ready=torch.profiler.tensorboard_trace_handler("./time")) as p:
        for _ in range(1000):         
            test_scaled_dot_product_attention()
    end_event.record()
    torch.cuda.current_stream().synchronize()
    print(base_name)
    total_e2e_time = start_event.elapsed_time(end_event)
    print(p.key_averages().table(sort_by="self_cpu_time_total"))
    key_averages = p.key_averages()
    time = 0
    for avg in key_averages:
        if avg.key == 'aten::matmul':  # 根据 key 找到你需要的操作
            time = avg.cuda_time
    
    times.append(time)
print(times)


Function Overview:
`TENSORIZATION` in the context of SIMD (Single Instruction, Multiple Data) is a technique that 
transforms scalar operations into vectorized operations to take advantage of the parallel processing 
capabilities of modern processors. By converting scalar computations (processing one element at a time) 
into tensorized or vectorized computations, SIMD instructions can process multiple data points 
simultaneously, improving throughput and reducing the overall computation time.

Application Scenario:
- Tensorization is widely used in deep learning frameworks to speed up matrix multiplications, 
  convolutions, and other tensor operations by leveraging SIMD. For example, 
  it can be used to vectorize the processing of large batches of input data, 
  improving performance on CPUs, GPUs, and other accelerators.
- SIMD-based tensorization can be applied to common linear algebra kernels such as 
matrix-vector multiplications (GEMV), matrix-matrix multiplications (GEMM), and vector dot products. 
    SIMD instructions accelerate these operations by processing multiple elements of vectors or 
    matrices in parallel.


Example1：
void exmaple1(float* output, float* input_1, float* input_2) {
    for (i = 0; i < 64; ++i) {
        for (j = 0; j < 64; ++j) {
            for (k = 0; k < 64; ++k) {
                output[i * 64 + j] += input_1[i * 64 + k] * input_2[k * 64 + j];
            }
        }
    }
}

// after: 
void exmaple1(float* output, float* input_1, float* input_2) {
    __bang_mlp(output, input_1, input_2, 64, 64);
}

Example2：
void exmaple2(float* output, half* input_1, half* input_2) {
    for (i = 0; i < 32; ++i) {
        for (j = 0; j < 64; ++j) {
            for (k = 0; k < 64; ++k) {
                output[i * 64 + j] += input_1[i * 64 + k] * input_2[k * 64 + j];
            }
        }
    }
}

// after: 
void exmaple1(float* output, half* input_1, half* input_2) {
    __bang_mlp(output, input_1, input_2, 32, 64);
}

Input code:

