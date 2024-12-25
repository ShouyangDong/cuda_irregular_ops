import csv
import ctypes
import glob
import os
import re

import numpy as np
import torch

from benchmark.utils import avgpool_np, maxpool_np, minpool_np
from benchmark.utils import run_cuda_compilation as run_compilation
from benchmark.utils import sumpool_np


def create_cuda_perf_func(cuda_file):
    """读取CUDA文件，插入事件以获取内核执行时间，并写回文件"""
    with open(cuda_file, "r") as f:
        content = f.read()

    # 正则表达式匹配内核调用的行，比如 add<<<numBlocks, blockSize>>>(d_A, d_B, d_C);
    # TODO(michael): solve the case if the kernel change into another line
    kernel_call_pattern = r"(\w+<<<.*?>>>\(.*?\);)"  # 匹配具有<<<...>>>的行

    # 查找所有匹配的行
    matches = re.findall(kernel_call_pattern, content)
    for match in matches:
        # 插入热身和事件相关代码
        event_code = f"""
// Warm-up the kernel by running it 10 times
for (int i = 0; i < 10; i++) {{
    {match}
}}

// Timing the kernel execution
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start, 0);

for (int i = 0; i < 1000; i++) {{
    {match}
}}
cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
milliseconds = milliseconds / 1000.0f;
printf("Kernel execution time: %f ms\\n", milliseconds);
cudaEventDestroy(start);
cudaEventDestroy(stop);
        """

        # 在内核调用之后插入时间计算代码
        with open(
            os.path.join(os.getcwd(), "benchmark/macro/cuda_macro.txt"), "r"
        ) as f:
            macro = f.read()
        content = content.replace('extern "C" void', 'extern "C" float')
        modified_kernel = macro + content.replace(match, event_code)

        # 在主函数的末尾插入 return milliseconds;
        main_pattern = r'extern "C" float \w+\s*\(.*?\)\s*\{'  # 允许函数参数
        main_match = re.search(
            main_pattern, modified_kernel, re.DOTALL
        )  # 使用re.DOTALL

        if main_match:
            # 找到主函数的结束位置，以替换最后一个 `}`
            last_brace_index = modified_kernel.rfind("}")
            if last_brace_index != -1:
                modified_kernel = (
                    modified_kernel[:last_brace_index]
                    + "return milliseconds;\n}"
                    + modified_kernel[last_brace_index + 1 :]
                )
        else:
            print("未找到主函数定义。请检查函数名和签名。")
            return

        # 将修改后的内容写回原文件
        with open(cuda_file.replace(".cu", "_bak.cu"), "w") as f:
            f.write(modified_kernel)


def perf_unary(name, shape, function, dtype="float32"):
    # 定义函数参数和返回类型

    function.restype = ctypes.c_float
    # 创建输入数组
    input_array = np.random.uniform(size=shape).astype(dtype)

    # 创建输出数组
    output_array = np.zeros_like(input_array)

    # 将输入数组和输出数组转换为C指针类型
    input_ptr = input_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_ptr = output_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    if name == "rmsnorm":
        function.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
        ]
        elapsed_time = function(input_ptr, output_ptr, *shape)
    elif name == "softmax":
        function.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
        ]
        elapsed_time = function(
            input_ptr, output_ptr, np.prod(shape[:-1]), shape[-1]
        )
    else:
        function.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        # 调用C函数
        elapsed_time = function(input_ptr, output_ptr, np.prod(shape))
    return elapsed_time


def perf_layernorm(shape1, shape2, function, dtype="float32"):
    # 定义函数参数和返回类型
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    function.restype = ctypes.c_float
    # 创建输入数组
    input_array = np.random.uniform(size=shape1).astype(dtype)
    gamma = np.random.uniform(size=shape2).astype(dtype)
    beta = np.random.uniform(size=shape2).astype(dtype)
    # 创建输出数组
    output_array = np.zeros_like(input_array)

    # 将输入数组和输出数组转换为C指针类型
    input_ptr = input_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    gamma_ptr = gamma.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    beta_ptr = beta.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_ptr = output_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    # 调用C函数
    elapsed_time = function(
        input_ptr, gamma_ptr, beta_ptr, output_ptr, *shape1
    )
    return elapsed_time


def perf_binary(name, shape_A, shape_B, shape_C, function, dtype="float32"):
    A = np.random.rand(*shape_A).astype("float32")
    B = np.random.rand(*shape_B).astype("float32")
    shape = [*shape_A, shape_B[-1]]
    # Convert the matrices to contiguous memory for ctypes
    A_ptr = A.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    B_ptr = B.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    function.restype = ctypes.c_float
    # Call the function with the matrices and dimensions
    result_ctypes = np.zeros(shape_C, dtype=np.float32)
    output_ptr = result_ctypes.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    if name == "add":
        function.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        elapsed_time = function(A_ptr, B_ptr, output_ptr, np.prod(shape_A))
    elif name in ["gemm", "bmm"]:
        # Convert the matrices to contiguous memory for ctypes
        A = np.random.rand(*shape_A).astype("float16")
        B = np.random.rand(*shape_B).astype("float16")
        shape = [*shape_A, shape_B[-1]]
        # Convert the matrices to contiguous memory for ctypes
        A_ptr = A.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
        B_ptr = B.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
        function.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        elapsed_time = function(
            output_ptr,
            A_ptr,
            B_ptr,
            *shape,
        )
    elif name in ["conv1d"]:
        # Convert the matrices to contiguous memory for ctypes
        A = np.random.rand(*shape_A).astype("float32")
        B = np.random.rand(*shape_B).astype("float32")
        # Convert the matrices to contiguous memory for ctypes
        A_ptr = A.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        B_ptr = B.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        function.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
        ]
        elapsed_time = function(
            output_ptr, A_ptr, B_ptr, shape_A[0], shape_A[0] - shape_B[0] + 1
        )
    elif name in ["gemv"]:
        # Convert the matrices to contiguous memory for ctypes
        A = np.random.rand(*shape_A).astype("float32")
        B = np.random.rand(*shape_B).astype("float32")
        # Convert the matrices to contiguous memory for ctypes
        A_ptr = A.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        B_ptr = B.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        function.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
        ]
        elapsed_time = function(
            output_ptr, A_ptr, B_ptr, shape_A[0], shape_A[1]
        )
    return elapsed_time


def perf_conv2d(name, shape_A, shape_B, shape_C, stride, function):
    function.restype = ctypes.c_float
    data_np = torch.rand(shape_A)
    kernel_np = torch.rand(shape_B)
    function.restype = ctypes.c_float
    # Convert the matrices to contiguous memory for ctypes
    input_ptr = data_np.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    kernel_ptr = kernel_np.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )
    result_ctypes = torch.zeros(shape_C)
    output_ptr = result_ctypes.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )
    if name == "conv2d":
        function.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        elapsed_time = function(
            output_ptr,
            input_ptr,
            kernel_ptr,
            shape_A[0],
            shape_A[1],
            shape_A[3],
            shape_B[0],
            shape_B[1],
            stride,
        )
    elif name == "conv2dnchw":
        function.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        elapsed_time = function(
            output_ptr,
            input_ptr,
            kernel_ptr,
            shape_A[0],
            shape_A[2],
            shape_A[1],
            shape_B[0],
            shape_B[2],
            stride,
        )
    elif name == "depthwiseconv":
        function.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        elapsed_time = function(
            input_ptr,
            kernel_ptr,
            output_ptr,
            *stride,
        )
    return elapsed_time


def perf_deformable(shape, function):
    N, M, D = shape[:3]
    Lq, L, P = shape[3:]
    shapes = torch.as_tensor(
        [[84, 117], [42, 59], [21, 30], [11, 15]], dtype=torch.long
    )
    level_start_index = torch.cat(
        (shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1])
    )
    S = sum([(H * W).item() for H, W in shapes])

    value = torch.rand(N, S, M, D) * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 2)
    attention_weights = torch.rand(N, Lq, M, L, P) + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(
        -2, keepdim=True
    )

    # 定义函数参数和返回类型
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    function.restype = ctypes.c_float

    # 创建输出数组
    output_array = np.zeros(
        (
            value.shape[0],
            sampling_locations.shape[1],
            value.shape[2] * value.shape[3],
        ),
        "float32",
    )

    # 将输入数组和输出数组转换为C指针类型
    value_ptr = value.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    shapes_ptr = (
        shapes.int().numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    )
    sampling_locations_ptr = sampling_locations.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )
    level_start_index_ptr = (
        level_start_index.int()
        .numpy()
        .ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    )
    attention_weights_ptr = attention_weights.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )
    output_ptr = output_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    # 调用C函数
    elapsed_time = function(
        value_ptr,
        shapes_ptr,
        level_start_index_ptr,
        sampling_locations_ptr,
        attention_weights_ptr,
        output_ptr,
    )
    return elapsed_time


def perf_pooling(name, shape, kernel, stride, function, dtype="float32"):
    input_array = torch.rand(*shape)
    # Calculate the result using numpy for comparison
    if name == "maxpool":
        output_np = maxpool_np(input_array, kernel + stride)
    elif name == "minpool":
        output_np = minpool_np(input_array, kernel + stride)
    elif name == "avgpool":
        output_np = avgpool_np(input_array, kernel + stride)
    elif name == "sumpool":
        output_np = sumpool_np(input_array, kernel + stride)
    output_array = torch.zeros(output_np.shape)
    # Convert the arrays to contiguous memory for ctypes
    input_ptr = input_array.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )
    output_ptr = output_array.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )
    np.prod(shape)
    np.prod(output_np.shape)
    # 定义函数参数和返回类型
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    function.restype = ctypes.c_float
    # Call the function with the matrices and dimensions
    elapsed_time = function(
        input_ptr,
        output_ptr,
        shape[0],
        shape[3],
        shape[1],
        kernel[0],
        stride[0],
    )
    return elapsed_time


def perf_scaled_dot_product_attention(shape, function, dtype="float32"):
    # 定义函数参数和返回类型
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    function.restype = ctypes.c_float
    # 创建输入数组
    input_array_1 = np.random.uniform(size=shape).astype(dtype)
    input_array_2 = np.random.uniform(size=shape).astype(dtype)
    input_array_3 = np.random.uniform(size=shape).astype(dtype)
    # 创建输出数组
    output_array = np.zeros_like(input_array_1)

    # 将输入数组和输出数组转换为C指针类型
    input_ptr_1 = input_array_1.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    input_ptr_2 = input_array_2.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    input_ptr_3 = input_array_3.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_ptr = output_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    # 调用C函数
    elapsed_time = function(
        input_ptr_1, input_ptr_2, input_ptr_3, output_ptr, *shape
    )
    return elapsed_time


def perf_pipeline(cuda_file):
    create_cuda_perf_func(cuda_file)
    backup_cuda_file = cuda_file.replace(".cu", "_bak.cu")
    so_name = cuda_file.replace(".cu", ".so")
    success, output = run_compilation(so_name, backup_cuda_file)
    print(output)


def benchmark(cuda_file):
    execution_time = 0
    base_name = os.path.basename(cuda_file)
    name = base_name.split("_")[0]
    perf_pipeline(cuda_file)
    lib = ctypes.CDLL(cuda_file.replace(".cu", ".so"))
    function = getattr(lib, name + "_kernel")
    if name == "add":
        shapes = base_name.split(".")[0]
        shape = [int(intg) for intg in shapes.split("_")[1:]]
        execution_time = perf_binary(name, shape, shape, shape, function)

    elif name in ["avgpool", "maxpool", "minpool", "sumpool"]:
        shape = base_name.split("_")[1:5]
        shape = [int(intg) for intg in shape]
        kernel_stride = base_name.split(".")[0].split("_")[5:]
        kernel_stride = [int(intg) for intg in kernel_stride]
        execution_time = perf_pooling(
            name, shape, kernel_stride[:2], kernel_stride[2:], function
        )

    elif name == "bmm":
        shapes = base_name.split(".")[0]
        shape = [int(intg) for intg in shapes.split("_")[1:]]
        batch_size, matrix_dim_i, matrix_dim_j, matrix_dim_k = shape
        shape_A = [batch_size, matrix_dim_i, matrix_dim_j]
        shape_B = [batch_size, matrix_dim_k, matrix_dim_j]
        shape_C = [batch_size, matrix_dim_i, matrix_dim_k]
        execution_time = perf_binary(name, shape_A, shape_B, shape_C, function)

    elif name == "gemm":
        shapes = base_name.split(".")[0]
        shape = [int(intg) for intg in shapes.split("_")[1:]]
        shape_A = [1, shape[0], shape[1]]
        shape_B = [1, shape[2], shape[1]]
        shape_C = [1, shape[0], shape[2]]
        execution_time = perf_binary(name, shape_A, shape_B, shape_C, function)

    elif name in ["sign", "relu", "sigmoid", "softmax", "rmsnorm", "gelu"]:
        shapes = base_name.split(".")[0]
        shape = [int(intg) for intg in shapes.split("_")[1:]]
        execution_time = perf_unary(name, shape, function)

    elif name == "conv2d":
        data_shape = base_name.split("_")[1:5]
        data_shape = [int(intg) for intg in data_shape]

        kernel_shape = base_name.split("_")[5:9]
        kernel_shape = [int(intg) for intg in kernel_shape]
        stride_h = stride_w = int(base_name.split("_")[9])
        pad_h = pad_w = int(base_name.split("_")[10].replace(".cu", ""))

        batch_size, input_height, input_width, input_channel = data_shape
        output_channel, kernel_height, kernel_width, _ = kernel_shape
        out_height = int(
            (input_height + np.sum(pad_h) - kernel_height) / stride_h + 1
        )
        out_width = int(
            (input_width + np.sum(pad_w) - kernel_width) / stride_w + 1
        )
        output_shape = [batch_size, out_height, out_width, output_channel]
        execution_time = perf_conv2d(
            name, data_shape, kernel_shape, output_shape, stride_h, function
        )

    elif name == "conv2dnchw":
        data_shape = base_name.split("_")[1:5]
        data_shape = [int(intg) for intg in data_shape]
        kernel_shape = base_name.split("_")[5:9]
        kernel_shape = [int(intg) for intg in kernel_shape]
        stride_h = stride_w = int(base_name.split(".")[0].split("_")[9])
        pad = int(base_name.split(".")[0].split("_")[10])
        batch_size, input_channel, input_height, input_width = data_shape
        output_channel, _, kernel_height, kernel_width = kernel_shape
        out_height = int(
            (input_height + np.sum(pad) - kernel_height) / stride_h + 1
        )
        out_width = int(
            (input_width + np.sum(pad) - kernel_width) / stride_w + 1
        )
        output_shape = [batch_size, output_channel, out_height, out_width]
        execution_time = perf_conv2d(
            name, data_shape, kernel_shape, output_shape, stride_h, function
        )

    elif name == "gemv":
        shapes = base_name.split(".")[0]
        shape = [int(intg) for intg in shapes.split("_")[1:]]
        kernel_shape = [shape[1]]
        output_shape = [shape[0]]
        execution_time = perf_binary(
            name, shape, kernel_shape, output_shape, function
        )

    elif name == "conv1d":
        shapes = base_name.split(".")[0]
        shape = [int(intg) for intg in shapes.split("_")[1:]]
        shape = [shape[1]]
        kernel_shape = [3]
        output_shape = [shape[0]]
        execution_time = perf_binary(
            name, shape, kernel_shape, output_shape, function
        )

    elif name == "depthwiseconv":
        shapes = base_name.split(".")[0]
        shape = [int(intg) for intg in shapes.split("_")[1:]]
        input_height, kernel_size, input_channels = (
            shape[0],
            shape[1],
            shape[2],
        )
        data_shape = [input_height, input_height, input_channels]
        kernel_shape = [kernel_size, kernel_size, input_channels]
        # Calculate the output tensor shape
        output_height = input_height - kernel_size + 1
        output_width = input_height - kernel_size + 1
        output_shape = [output_height, output_width, input_channels]
        execution_time = perf_conv2d(
            name, data_shape, kernel_shape, output_shape, shape, function
        )

    elif name == "deformable":
        shapes = base_name.split(".")[0]
        shape = [int(intg) for intg in shapes.split("_")[1:]]
        execution_time = perf_deformable(shape, function)

    elif name == "mha":
        shapes = base_name.split(".")[0]
        shape = [int(intg) for intg in shapes.split("_")[1:]]
        perf_scaled_dot_product_attention(shape, function)

    elif name == "layernorm":
        shapes = base_name.split(".")[0]
        shape = [int(intg) for intg in shapes.split("_")[1:]]
        execution_time = perf_layernorm(shape, [shape[-1]], function)

    else:
        print("Undefined file: ", cuda_file)
        return 0

    os.remove(cuda_file.replace(".cu", "_bak.cu"))
    os.remove(cuda_file.replace(".cu", ".so"))
    return execution_time


if __name__ == "__main__":
    files = glob.glob(
        os.path.join(os.getcwd(), "benchmark/data/cuda_code_test/*.cu")
    )

    table = []
    times = []
    table.append(files)
    for file in files:
        execution_time = benchmark(file)
        times.append(execution_time)

    table.append(times)

    # 转置数据
    transposed_data = list(zip(*table))

    # 添加标题行
    header = ["file", "time(ms)"]
    transposed_data.insert(0, header)

    # 保存为CSV文件
    with open("benchmark/perf/cuda_output.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(transposed_data)
