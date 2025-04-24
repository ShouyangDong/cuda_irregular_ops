import argparse
import ctypes
import os
import subprocess
from ctypes import CDLL

import torch  # 引入 PyTorch

from benchmark.utils import run_dlboost_compilation as run_compilation


def ref_program(x):
    return torch.softmax(x, dim=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="the source file")
    args = parser.parse_args()
    base_name = os.path.basename(args.file)
    name = "softmax"
    shapes = base_name.split(".")[0]
    shape = [int(intg) for intg in shapes.split("_")[1:]]

    so_name = args.file.replace(".cpp", ".so")
    with open(args.file, "r") as f:
        code = f.read()

    with open(
        os.path.join(os.getcwd(), "benchmark/macro/cpp_macro.txt"), "r"
    ) as f:
        macro = f.read()
    code = macro + code

    file_name = args.file.replace(
        base_name.replace(".cpp", ""), base_name + "_bak.cpp"
    )
    with open(file_name, mode="w") as f:
        f.write(code)

    success, output = run_compilation(so_name, file_name)
    os.remove(file_name)

    # 加载 C 库
    lib = CDLL(os.path.join(os.getcwd(), so_name))
    function = getattr(lib, name + "_kernel")

    # 定义函数参数和返回类型
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    function.restype = None

    # 创建输入数组
    dtype = "float32"
    input_array = torch.randn(shape)
    expected_output = ref_program(input_array)

    # 创建输出数组
    output_array = torch.zeros(shape)

    # 将输入数组和输出数组转换为 C 指针类型
    input_ptr = input_array.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )
    output_ptr = output_array.numpy().ctypes.data_as(
        ctypes.POINTER(ctypes.c_float)
    )

    # 调用 C 函数
    function(input_ptr, output_ptr)

    # 验证结果
    torch.allclose(
        output_array,
        expected_output,
        rtol=1e-03,
        atol=1e-03,
        equal_nan=True,
    )

    print("验证通过！")
    result = subprocess.run(["rm", so_name])
