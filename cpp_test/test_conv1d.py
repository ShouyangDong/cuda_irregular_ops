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


# Define the input array and kernel
input_array = np.array([1.0, 2.0, 1.0, 3.0, 0.0, 1.0, 2.0]).astype(np.float32)
kernel = np.array([0.5, 1.0, 0.5]).astype(np.float32)

# Calculate the output size
output_size = input_array.size - kernel.size + 1

# Create an empty output array
output_ctypes = np.zeros(output_size, dtype=np.float32)

# Convert the arrays to contiguous memory for ctypes
input_ptr = input_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
kernel_ptr = kernel.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
output_ptr = output_ctypes.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

# Calculate the result using numpy for comparison
output_np = np.convolve(input_array, kernel, mode='valid')

file_name = "conv_1d.cpp"
so_name = "conv_1d.so"
# Load the shared library with the batch matrix multiplication function
success, output = run_compilation(so_name, file_name)
# # os.remove(file_name)
lib = ctypes.CDLL(os.path.join(os.getcwd(), so_name))
function = getattr(lib, "conv_1d_kernel")
# 定义函数参数和返回类型
function.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
]
function.restype = None
# Call the function with the matrices and dimensions
function(output_ptr, input_ptr, kernel_ptr)
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
print("Successful")