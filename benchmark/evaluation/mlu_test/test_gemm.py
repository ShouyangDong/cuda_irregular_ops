import argparse
import ctypes
import os
import subprocess

import numpy as np

from benchmark.template.mlu_host_template import create_bang_func
from benchmark.utils import run_mlu_compilation as run_compilation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="the source file")
    args = parser.parse_args()
    base_name = os.path.basename(args.file)
    shapes = base_name.split(".")[0]
    shape = [int(intg) for intg in shapes.split("_")[1:]]
    # Generate random matrices for testing
    # Define the input matrix A and vector x
    A = np.ones((shape[0], shape[1])).astype(np.float16)
    x = np.ones((shape[1], shape[2])).astype(np.float16)
    name = base_name.split("_")[0]
    # Create an empty vector y
    y_ctypes = np.zeros((shape[0], shape[2]), dtype=np.float32)

    # Convert the matrices to contiguous memory for ctypes
    A_ptr = A.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
    x_ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
    y_ptr = y_ctypes.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Perform gemm using numpy
    y_np = np.matmul(A, x)

    # Load the shared library with the batch matrix multiplication function
    so_name = args.file.replace(".mlu", ".so")
    file_name = create_bang_func(args.file, op_type="matmul")
    # Load the shared library with the batch matrix multiplication function
    success, output = run_compilation(so_name, file_name)

    os.remove(file_name)
    lib = ctypes.CDLL(os.path.join(os.getcwd(), so_name))
    function = getattr(lib, name + "_kernel")
    # 定义函数参数和返回类型
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    function.restype = None
    # Call the function with the matrices and dimensions
    function(y_ptr, A_ptr, x_ptr, *shape)
    # Check if the results match
    np.testing.assert_allclose(
        y_ctypes,
        y_np,
        rtol=1e-03,
        atol=1e-03,
        equal_nan=True,
        err_msg="",
        verbose=True,
    )
    print("验证通过！")
    result = subprocess.run(["rm", so_name])
