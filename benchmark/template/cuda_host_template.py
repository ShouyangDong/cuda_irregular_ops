import re
from string import Template


def create_cuda_host_func(file_name):
    with open(file_name, "r") as f:
        original_function = f.read()

    # 提取函数名、参数列表、launch_bounds里的block大小
    function_signature_pattern = r"__global__ void\s*(?:__launch_bounds__\((\d+)(?:,\s*\d+)?\))?\s*(\w+)\(([^)]*)\)"
    match = re.search(function_signature_pattern, original_function, re.DOTALL)
    if not match:
        raise ValueError("Could not find function signature.")

    thread_num = match.group(1)
    kernel_name = match.group(2)
    param_list_str = match.group(3)

    if thread_num is None:
        thread_num = "256"  # 默认256
    else:
        thread_num = thread_num.strip()

    params = [param.strip() for param in param_list_str.split(",")]
    param_names = [
        param.split()[-1].replace("*", "").replace("__restrict__", "").strip()
        for param in params
    ]

    # 生成 device端变量名
    device_vars = [f"d_{name}" for name in param_names]

    # 生成host端参数
    host_param_list = ", ".join(params + ["int size"])

    # 分配device内存
    device_malloc = "\n  ".join(
        [
            f"cudaMalloc(&{dev_var}, size * sizeof(float));"
            for dev_var in device_vars
        ]
    )

    # Host->Device memcpy
    memcpy_htod = "\n  ".join(
        [
            f"cudaMemcpy({dev_var}, {param}, size * sizeof(float), cudaMemcpyHostToDevice);"
            for dev_var, param in zip(device_vars, param_names[:-1])
        ]
    )

    # Device->Host memcpy
    memcpy_dtoh = f"cudaMemcpy({param_names[-1]}, {device_vars[-1]}, size * sizeof(float), cudaMemcpyDeviceToHost);"

    # cudaFree
    cuda_free = "\n  ".join(
        [f"cudaFree({dev_var});" for dev_var in device_vars]
    )

    # 动态替换模板
    host_func_template = Template(
        """
extern "C" void ${kernel_name}_kernel(${host_param_list}) {
  float *${device_vars};

  ${device_malloc}

  ${memcpy_htod}

  dim3 blockSize(${thread_num});
  dim3 numBlocks((size + ${thread_num} - 1) / ${thread_num});

  ${kernel_name}<<<numBlocks, blockSize>>>(${device_call_args});

  ${memcpy_dtoh}

  ${cuda_free}
}
"""
    )

    new_code = host_func_template.substitute(
        kernel_name=kernel_name,
        host_param_list=host_param_list,
        device_vars=", ".join(device_vars),
        device_malloc=device_malloc,
        memcpy_htod=memcpy_htod,
        thread_num=thread_num,
        device_call_args=", ".join(device_vars),
        memcpy_dtoh=memcpy_dtoh,
        cuda_free=cuda_free,
    )
    device_code = original_function.split("extern")[0].strip()
    new_code = device_code + "\n" + new_code

    print("[INFO]***************new_code: ", new_code)
    # 保存
    output_file = file_name.replace(".cu", "_host.cu")
    with open(output_file, "w") as f:
        f.write(new_code)

    return output_file


if __name__ == "__main__":
    create_cuda_host_func("benchmark/data/cuda_code_test/add_320.cu")
