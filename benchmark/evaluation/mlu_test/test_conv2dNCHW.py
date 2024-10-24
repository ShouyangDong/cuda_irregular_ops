import argparse
import ctypes
import os
import subprocess

import numpy as np


def run_compilation(so_name, file_name):
    try:
        output = subprocess.run(
            ["nvcc", "-shared", "-Xcompiler", "-fPIC", "-o", so_name, file_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            check=True,
            text=True,
            timeout=15,
        )
        return True, output
    except subprocess.CalledProcessError as e:
        return False, e.output


def cpu_conv(input_tensor, kernel, stride, pad=0):
    # 获取输入张量和卷积核的维度
    N, C, H, W = input_tensor.shape
    out_channels, in_channels, kernel_height, kernel_width = kernel.shape

    # 计算输出张量的空间维度
    out_height = (H - kernel_height) // stride + 1
    out_width = (W - kernel_width) // stride + 1

    # 初始化输出张量
    output_tensor = np.zeros((N, out_channels, out_height, out_width))

    # 执行卷积操作
    for n in range(N):
        for out_channel in range(out_channels):
            for in_channel in range(in_channels):
                for i in range(0, out_height):
                    for j in range(0, out_width):
                        h_start = i * stride
                        h_end = h_start + kernel_height
                        w_start = j * stride
                        w_end = w_start + kernel_width
                        output_tensor[n, out_channel, i, j] += np.sum(
                            input_tensor[n, in_channel, h_start:h_end, w_start:w_end]
                            * kernel[out_channel, in_channel]
                        )

    return output_tensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="the source file")

    args = parser.parse_args()
    base_name = os.path.basename(args.file)

    name = base_name.split("_")[0]
    data_shape = base_name.split("_")[1:5]

    data_shape = [int(intg) for intg in data_shape]

    kernel_shape = base_name.split("_")[5:9]
    kernel_shape = [int(intg) for intg in kernel_shape]
    stride_h = stride_w = int(base_nam.split(".")[0].split("_")[9])
    pad = int(base_name.split(".")[0].split("_")[10])
    dtype = "float32"
    wtype = "float32"

    # generate data
    data_np = np.random.uniform(low=1.0, high=2.0, size=data_shape).astype(dtype)
    kernel_np = np.random.uniform(low=1.0, high=2.0, size=kernel_shape).astype(dtype)
    # cpu compute
    result_cpu = cpu_conv(data_np, kernel_np, stride_h, pad)

    # Load the shared library with the conv2d function
    so_name = args.file.replace(".mlu", ".so")
    with open(args.file, "r") as f:
        code = f.read()
        f.close()

    with open("./macro/mlu_macro.txt", "r") as f:
        macro = f.read()
        f.close()
    code = macro + code

    file_name = args.file.replace(base_name.replace(".mlu", ""), base_name + "_bak.mlu")
    with open(file_name, mode="w") as f:
        f.write(code)
        f.close()
    success, output = run_compilation(so_name, file_name)
    os.remove(file_name)

    lib = ctypes.CDLL(os.path.join(os.getcwd(), so_name))
    function = getattr(lib, name + "_kernel")
    # 定义函数参数和返回类型
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    function.restype = None
    # Convert the matrices to contiguous memory for ctypes
    input_ptr = data_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    kernel_ptr = kernel_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    result_ctypes = np.zeros(result_cpu.shape, dtype=np.float32)
    output_ptr = result_ctypes.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    # Call the function with the matrices and dimensions
    function(input_ptr, kernel_ptr, output_ptr)
    # Check if the results match
    np.testing.assert_allclose(
        output_ctypes,
        output_np,
        rtol=1e-03,
        atol=1e-03,
        equal_nan=True,
        err_msg="",
        verbose=True,
    )
    print("验证通过！")
    result = subprocess.run(["rm", so_name])
