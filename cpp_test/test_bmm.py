import numpy as np
import ctypes
import subprocess
import os
import argparse

def run_compilation(so_name, file_name):
    try:
        output = subprocess.run(
            ["g++", "-shared", "-fPIC", "-o", so_name, file_name],
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

# Define the batch matrix multiplication function using numpy
def batch_matmul(A, B):
    return np.matmul(A, B)


# Generate random matrices for testing
batch_size = 3
matrix_dim_i = 4
matrix_dim_j = 5
matrix_dim_k = 6
A = np.random.rand(batch_size, matrix_dim_i, matrix_dim_j).astype("float32")
B = np.random.rand(batch_size, matrix_dim_j, matrix_dim_k).astype("float32")

# Perform batch matrix multiplication using numpy
result_np = batch_matmul(A, B)

# Convert the matrices to contiguous memory for ctypes
A_ptr = A.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
B_ptr = B.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
file_name = "bmm.cpp"
so_name = "bmm.so"
# Load the shared library with the batch matrix multiplication function
success, output = run_compilation(so_name, file_name)
# # os.remove(file_name)
lib = ctypes.CDLL(os.path.join(os.getcwd(), so_name))
function = getattr(lib, "bmm_kernel")
# 定义函数参数和返回类型
function.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
]
function.restype = None
# Call the function with the matrices and dimensions
result_ctypes = np.zeros((batch_size, matrix_dim_i, matrix_dim_k), dtype=np.float32)
output_ptr = result_ctypes.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
function(output_ptr, A_ptr, B_ptr)
# Check if the results match
np.testing.assert_allclose(
    result_ctypes,
    result_np,
    rtol=1e-03,
    atol=1e-03,
    equal_nan=True,
    err_msg="",
    verbose=True,
)
print("Successful")