import argparse
import ctypes
import os
import subprocess
from ctypes import CDLL

import numpy as np


def run_compilation(so_name, file_name):
    try:
        output = subprocess.run(
            [
                "g++",
                "-shared",
                "-fPIC",
                "-march=icelake-server",
                "-O3",
                file_name,
                "-o",
                so_name,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            check=True,
            # text=True,
            timeout=15,
        )
        return True, output
    except subprocess.CalledProcessError as e:
        return False, e.output


def ref_program(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    x_normalized = (x - mean) / (std + eps)
    out = gamma * x_normalized + beta
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="the source file")
    args = parser.parse_args()
    base_name = os.path.basename(args.file)
    name = "layer_norm"
    shapes = base_name.split(".")[0]
    shape = [int(intg) for intg in shapes.split("_")[1:]]

    so_name = args.file.replace(".cpp", ".so")
    with open(args.file, "r") as f:
        code = f.read()
        f.close()

    with open("./macro/dlboost_macro.txt", "r") as f:
        macro = f.read()
        f.close()
    code = macro + code

    file_name = args.file.replace(base_name.replace(".cpp", ""), base_name + "_bak.cpp")
    with open(file_name, mode="w") as f:
        f.write(code)
        f.close()

    success, output = run_compilation(so_name, file_name)
    os.remove(file_name)
    success, output = run_compilation(so_name, file_name)
    lib = CDLL(os.path.join(os.getcwd(), so_name))
    function = getattr(lib, name + "_kernel")
    # 定义函数参数和返回类型
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    function.restype = None
    # 创建输入数组
    dtype = "float32"
    input_array = np.random.uniform(size=shape).astype(dtype)
    gamma_array = np.random.uniform(size=shape[-1:]).astype(dtype)
    beta_array = np.random.uniform(size=shape[-1:]).astype(dtype)
    expected_output = ref_program(input_array, gamma_array, beta_array)

    # 创建输出数组
    output_array = np.zeros_like(input_array)

    # 将输入数组和输出数组转换为C指针类型
    input_ptr = input_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    gamma_ptr = gamma_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    beta_ptr = beta_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_ptr = output_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    # 调用C函数
    function(input_ptr, gamma_ptr, beta_ptr, output_ptr)
    # 验证结果

    np.testing.assert_allclose(
        output_array,
        expected_output,
        rtol=1e-03,
        atol=1e-03,
        equal_nan=True,
        err_msg="",
        verbose=True,
    )
    print("验证通过！")
    result = subprocess.run(["rm", so_name])
