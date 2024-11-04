import argparse
import ctypes
import os
import subprocess

import numpy as np


def run_compilation(so_name, file_name):
    try:
        output = subprocess.run(
            ["cncc", "-shared", "--bang-mlu-arch=mtp_592", "-fPIC", "-o", so_name, file_name],
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="the source file")
    args = parser.parse_args()
    base_name = os.path.basename(args.file)
    shapes = base_name.split(".")[0]
    shape = [int(intg) for intg in shapes.split("_")[1:]]
    # Generate random matrices for testing
    # Define the input matrix A and vector x
    A = np.random.rand(shape[0], shape[1]).astype(np.float32)
    x = np.random.rand(shape[1], shape[2]).astype(np.float32)

    # Create an empty vector y
    y_ctypes = np.zeros((shape[0], shape[2]), dtype=np.float32)

    # Convert the matrices to contiguous memory for ctypes
    A_ptr = A.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    x_ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    y_ptr = y_ctypes.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Perform gemm using numpy
    y_np = np.matmul(A, x)

    # Load the shared library with the batch matrix multiplication function
    so_name = args.file.replace(".mlu", ".so")
    with open(args.file, "r") as f:
        code = f.read()
        f.close()

    with open(os.path.join(os.getcwd(), "benchmark/macro/mlu_macro.txt"), "r") as f:
        macro = f.read()
        f.close()
    code = macro + code

    file_name = args.file.replace(base_name.replace(".mlu", ""), base_name + "_bak.mlu")
    with open(file_name, mode="w") as f:
        f.write(code)
        f.close()
    # Load the shared library with the batch matrix multiplication function
    success, output = run_compilation(so_name, file_name)

    os.remove(file_name)
    lib = ctypes.CDLL(os.path.join(os.getcwd(), so_name))
    function = getattr(lib, "gemm_kernel")
    # 定义函数参数和返回类型
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    function.restype = None
    # Call the function with the matrices and dimensions
    function(y_ptr, A_ptr, x_ptr)
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
