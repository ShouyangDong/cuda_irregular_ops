import re
from string import Template


def infer_grid_dim_from_kernel(kernel_code: str) -> str:
    use_x = bool(re.search(r"\bblockIdx\.x\b", kernel_code))
    use_y = bool(re.search(r"\bblockIdx\.y\b", kernel_code))
    use_z = bool(re.search(r"\bblockIdx\.z\b", kernel_code))

    if use_z:
        return "3D"
    elif use_y:
        return "2D"
    elif use_x:
        return "1D"
    else:
        return "Unknown"


def create_cuda_func(file_name, op_type="ewise"):
    with open(file_name, "r") as f:
        original_function = f.read()

    # Regular expression to extract function signature
    function_signature_pattern = r"__global__ void\s*(?:__launch_bounds__\((\d+)(?:,\s*\d+)?\))?\s*(\w+)\(([^)]*)\)"
    match = re.search(function_signature_pattern, original_function, re.DOTALL)
    if not match:
        raise ValueError("Could not find CUDA kernel signature.")

    thread_num = match.group(1)
    kernel_name = match.group(2)
    param_list_str = match.group(3)

    if thread_num is None:
        thread_num = "1024"  # Default value

    params = [param.strip() for param in param_list_str.split(",")]
    params = [var.replace("__restrict__ ", "").strip() for var in params]
    params = [var.replace("const ", "").strip() for var in params]
    param_list = ", ".join(params)
    param_names = [
        param.split()[-1].replace("*", "").replace("__restrict__", "").strip()
        for param in params
    ]

    device_vars = [f"{name}_cuda" for name in param_names]

    # Memory allocation and copy operations
    device_memory_alloc = []
    memcpy = []
    size = None
    if op_type == "ewise":
        size = "size"
        for param in params:
            name = param.split("*")[1]
            device_memory_alloc.append(param + "_cuda;\n")
            device_memory_alloc.append(
                f"cudaMalloc((void**)&{name}_cuda, {size} * sizeof(float));\n"
            )

        for param in params[:-1]:
            name = param.split("*")[1]
            memcpy.append(
                f"cudaMemcpy({name}_cuda, {name}, {size} * sizeof(float), cudaMemcpyHostToDevice);\n"
            )
        # Copy back
        name = params[-1].split("*")[1]
        memcpy_back = f"cudaMemcpy({name}, {name}_cuda, {size} * sizeof(float), cudaMemcpyDeviceToHost);\n"
    elif op_type == "pool":
        size = ["size1", "size2"]
        for i, param in enumerate(params):
            name = param.split("*")[1]
            device_memory_alloc.append(param + "_cuda;\n")
            device_memory_alloc.append(
                f"cudaMalloc((void**)&{name}_cuda, {size[i]} * sizeof(float));\n"
            )

        for i, param in enumerate(params[:-1]):
            name = param.split("*")[1]
            memcpy.append(
                f"cudaMemcpy({name}_cuda, {name}, {size[i]} * sizeof(float), cudaMemcpyHostToDevice);\n"
            )
        # Copy back
        name = params[-1].split("*")[1]
        memcpy_back = f"cudaMemcpy({name}, {name}_cuda, {size[-1]} * sizeof(float), cudaMemcpyDeviceToHost);\n"
    elif op_type == "matmul":
        size = ["size1", "size2", "size3"]
        for i, param in enumerate(params):
            name = param.split("*")[1]
            dtype = param.split("*")[0]
            device_memory_alloc.append(param + "_cuda;\n")
            device_memory_alloc.append(
                f"cudaMalloc((void**)&{name}_cuda, {size[i]} * sizeof({dtype}));\n"
            )

        for i, param in enumerate(params[:-1]):
            name = param.split("*")[1]
            dtype = param.split("*")[0]
            memcpy.append(
                f"cudaMemcpy({name}_cuda, {name}, {size[i]} * sizeof({dtype}), cudaMemcpyHostToDevice);\n"
            )
        # Copy back
        name = params[-1].split("*")[1]
        dtype = params[-1].split("*")[0]
        memcpy_back = f"cudaMemcpy({name}, {name}_cuda, size3 * sizeof({dtype}), cudaMemcpyDeviceToHost);\n"

    elif op_type == "layer_norm":
        size = ["size1", "size2"]
        for i, param in enumerate(params):
            name = param.split("*")[1]
            device_memory_alloc.append(param + "_cuda;\n")
            if i == 1 or i == 2:
                device_memory_alloc.append(
                    f"cudaMalloc((void**)&{name}_cuda, size2 * sizeof(float));\n"
                )
            else:
                device_memory_alloc.append(
                    f"cudaMalloc((void**)&{name}_cuda, size1 * sizeof(float));\n"
                )
        for i, param in enumerate(params[:-1]):
            name = param.split("*")[1]
            if i == 1 or i == 2:
                memcpy.append(
                    f"cudaMemcpy({name}_cuda, {name}, size2 * sizeof(float), cudaMemcpyHostToDevice);\n"
                )
            else:
                memcpy.append(
                    f"cudaMemcpy({name}_cuda, {name}, size1 * sizeof(float), cudaMemcpyHostToDevice);\n"
                )
        # Copy back
        name = params[-1].split("*")[1]
        memcpy_back = f"cudaMemcpy({name}, {name}_cuda, size1 * sizeof(float), cudaMemcpyDeviceToHost);\n"

    # Infer grid dimensions from kernel code
    device_code = original_function.split("extern")[0]
    grid_dim = infer_grid_dim_from_kernel(device_code)
    if isinstance(size, list):
        size_list = ", ".join(
            arg for arg in ["int " + string for string in size]
        )
    else:
        size_list = "int size"
    if grid_dim == "1D":
        numblocks_define = f"dim3 numBlocks(256);"
    elif grid_dim == "2D":
        numblocks_define = f"dim3 numBlocks(256, 256); // 2D Grid, second dimension default to 8"
    elif grid_dim == "3D":
        numblocks_define = (
            f"dim3 numBlocks(256, 256, 256); // 3D Grid, default 4x4"
        )
    else:
        numblocks_define = f"dim3 numBlocks(1); // Unknown, default to 1D"

    # Create host function template
    host_func_template = Template(
        """
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <mma.h>
#include <cuda_fp16.h>
#include <stdlib.h>

using namespace nvcuda;

// Original Kernel
${original_function}

extern "C" void ${kernel_name}_kernel(${param_list}, ${size_list}) {
    ${memcpy_alloc_list}
    ${memcpy_htod}

    dim3 blockSize(${thread_num});
    ${numblocks_define}
    ${kernel_name}<<<numBlocks, blockSize>>>(${called_param_list});

    ${memcpy_dtoh}
    ${cuda_free}
}
"""
    )
    memcpy_alloc_list = "    ".join(alloc for alloc in device_memory_alloc)
    new_code = host_func_template.substitute(
        kernel_name=kernel_name,
        original_function=device_code.strip(),
        param_list=param_list,
        memcpy_htod="\n    ".join(memcpy),
        thread_num=thread_num,
        numblocks_define=numblocks_define,
        called_param_list=", ".join(device_vars),
        memcpy_dtoh=memcpy_back,
        cuda_free="\n    ".join(
            [f"cudaFree({dev}_cuda);" for dev in param_names]
        ),
        size_list=size_list,
        memcpy_alloc_list=memcpy_alloc_list,
    )

    output_file = file_name.replace(".cu", "_bak.cu")
    with open(output_file, "w") as f:
        f.write(new_code)

    return output_file


if __name__ == "__main__":
    create_cuda_perf_func("benchmark/data/cuda_code_test/sign_5_128.cu")
