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


# Define the input tensor, kernel, and parameters
input_tensor = np.random.rand(3, 5, 5).astype(np.float32)
kernel = np.random.rand(3, 3).astype(np.float32)
input_channels, input_height, input_width = input_tensor.shape
kernel_size = kernel.shape[0]

# Calculate the output tensor shape
output_height = input_height - kernel_size + 1
output_width = input_width - kernel_size + 1

# Create an empty output tensor
output_ctypes = np.zeros((input_channels, output_height, output_width), dtype=np.float32)

# Convert the arrays to contiguous memory for ctypes
input_ptr = input_tensor.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
kernel_ptr = kernel.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
output_ptr = output_ctypes.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
# Calculate the result using numpy for comparison
output_np = np.zeros((input_channels, output_height, output_width), dtype=np.float32)
for c in range(input_channels):
    for i in range(output_height):
        for j in range(output_width):
            output_np[c, i, j] = np.sum(input_tensor[c, i:i+kernel_size, j:j+kernel_size] * kernel[c])

    
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
print("验证通过！")