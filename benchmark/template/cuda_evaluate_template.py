import re
from string import Template


def create_cuda_perf_func(file_name):
    with open(file_name, "r") as f:
        original_function = f.read()

    # 正则匹配提取 kernel
    function_signature_pattern = r"__global__ void\s*(?:__launch_bounds__\((\d+)(?:,\s*\d+)?\))?\s*(\w+)\(([^)]*)\)"
    match = re.search(function_signature_pattern, original_function, re.DOTALL)
    if not match:
        raise ValueError("Could not find CUDA kernel signature.")

    thread_num = match.group(1)
    kernel_name = match.group(2)
    param_list_str = match.group(3)

    if thread_num is None:
        thread_num = "256"  # 默认值

    params = [param.strip() for param in param_list_str.split(",")]
    param_list = ", ".join(params)
    param_names = [
        param.split()[-1].replace("*", "").replace("__restrict__", "").strip()
        for param in params
    ]

    device_vars = [f"d_{name}" for name in param_names]

    # 构建 Host -> Device malloc/memcpy
    malloc_list = "\n    ".join(
        [f"cudaMalloc(&{dev}, size * sizeof(float));" for dev in device_vars]
    )
    memcpy_list = "\n    ".join(
        [
            f"cudaMemcpy({dev}, {name}, size * sizeof(float), cudaMemcpyHostToDevice);"
            for dev, name in zip(device_vars, param_names[:-1])
        ]
    )
    memcpy_back = f"cudaMemcpy({param_names[-1]}, {device_vars[-1]}, size * sizeof(float), cudaMemcpyDeviceToHost);"
    cuda_free = "\n    ".join([f"cudaFree({dev});" for dev in device_vars])

    called_param_list = ", ".join(device_vars)

    host_func_template = Template(
        """
#include <cuda_runtime.h>
#include <stdio.h>

// Original Kernel
${original_function}

extern "C" float timed_${kernel_name}_kernel(${param_list}, int size) {
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    ${device_malloc}
    ${memcpy_htod}

    dim3 blockSize(${thread_num});
    dim3 numBlocks((size + ${thread_num} - 1) / ${thread_num});

    // Warmup
    for (int i = 0; i < 10; ++i) {
        ${kernel_name}<<<numBlocks, blockSize>>>(${called_param_list});
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < 1000; ++i) {
        ${kernel_name}<<<numBlocks, blockSize>>>(${called_param_list});
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, end);
    printf("Total Time: %.3f ms\\n", ms);

    ${memcpy_dtoh}
    ${cuda_free}

    cudaEventDestroy(start);
    cudaEventDestroy(end);
    return ms;
}
"""
    )
    device_code = original_function.split("extern")[0]
    new_code = host_func_template.substitute(
        kernel_name=kernel_name,
        original_function=device_code.strip(),
        param_list=param_list,
        device_malloc=malloc_list,
        memcpy_htod=memcpy_list,
        thread_num=thread_num,
        called_param_list=called_param_list,
        memcpy_dtoh=memcpy_back,
        cuda_free=cuda_free,
    )
    print("[INFO]***************new_code: ", new_code)
    # 保存
    output_file = file_name.replace(".cu", "_timed.cu")
    with open(output_file, "w") as f:
        f.write(new_code)

    return output_file


if __name__ == "__main__":
    create_cuda_perf_func("benchmark/data/cuda_code_test/add_320.cu")
