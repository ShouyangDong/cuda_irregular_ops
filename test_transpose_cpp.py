import numpy as np
from ctypes import CDLL, c_void_p, c_double, c_int
import subprocess
import glob
import os
import ctypes
import argparse


def run_compilation(so_name, file_name):
    try:
        output = subprocess.run(
            ["g++", "-shared", "-fPIC", "-o", so_name, file_name],
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


def softmax(x):
    # 对最后一个维度进行softmax操作
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="the source file")
    parser.add_argument('--conversion', action='store_true', help='A boolean argument')
    args = parser.parse_args()
    base_name = os.path.basename(args.file)
    name = base_name.split("_")[0]
    shape = base_name.split("_")[1:-1]
    shape = [int(intg) for intg in shape]
    so_name = args.file.replace(".cpp", ".so")
    dtype = base_name.split("_")[-1].replace(".cpp", "")
    with open(args.file, "r") as f:
        code = f.read()
        f.close()

    with open("./macro/cpp_macro.txt", "r") as f:
        macro = f.read()
        f.close()
    code = macro + code
    code = code.replace("kernel0(", "kernel(")
    file_name = args.file.replace(base_name.replace(".cpp", ""), base_name + "_bak.cpp")
    with open(file_name, mode="w") as f:
        f.write(code)
        f.close()


    success, output = run_compilation(so_name, file_name)

    os.remove(file_name)
    lib = CDLL(so_name)
    function = getattr(lib, name + "_kernel")
    # 定义函数参数和返回类型
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    function.restype = None
    # 创建输入数组
    input_array = np.random.uniform(size=shape).astype(dtype)
    expected_output = softmax(input_array)

    # 创建输出数组
    output_array = np.zeros_like(input_array)

    # 将输入数组和输出数组转换为C指针类型
    input_ptr = input_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_ptr = output_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    # 调用C函数
    function(input_ptr, output_ptr)
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
    import time 
    t1 = time.time()
    for i in range(20):
        function(input_ptr, output_ptr)    
    t2 = time.time()
    cost = (t2 - t1) / 20.0 * 1e3
    with open(args.file.replace(".cpp", ".txt"), mode="w") as f:
        f.write(str(cost))
        f.close()
    result = subprocess.run(["rm", so_name])