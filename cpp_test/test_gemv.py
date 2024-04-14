import numpy as np
import ctypes
import subprocess
import os

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


# Generate random matrices for testing
# Define the input matrix A and vector x
A = np.random.rand(3, 4).astype(np.float32)
x = np.random.rand(4).astype(np.float32)

# Create an empty vector y
y_ctypes = np.zeros(3, dtype=np.float32)

# Convert the matrices to contiguous memory for ctypes
A_ptr = A.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
x_ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
y_ptr = y_ctypes.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

# Perform gemv using numpy
y_np = np.matmul(A, x)

file_name = "gemv.cpp"
so_name = "gemv.so"
# Load the shared library with the batch matrix multiplication function
success, output = run_compilation(so_name, file_name)
# # os.remove(file_name)
lib = ctypes.CDLL(os.path.join(os.getcwd(), so_name))
function = getattr(lib, "gemv_kernel")
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
print("Successful")