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
            timeout=15,
        )
        return True, output
    except subprocess.CalledProcessError as e:
        return False, e.output

if __name__ == "__main__":
    name = "transpose"
    shape = [8192, 8192]
    file_name = "8192_8192_transpose.cpp"
    so_name = "8192_8192_transpose.so"
    
    success, output = run_compilation(so_name, file_name)
    lib = CDLL(so_name)
    function = getattr(lib, name + "_kernel")
    # 定义函数参数和返回类型
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    function.restype = None
    # 创建输入数组
    dtype = "float32"
    input_array = np.random.uniform(size=shape).astype(dtype)
    expected_output = np.transpose(input_array)

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
    print("[INFO]*******cost: ", cost)
