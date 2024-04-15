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
            text=True,
            timeout=15,
        )
        return True, output
    except subprocess.CalledProcessError as e:
        return False, e.output

def depthwise_conv2d(input, w):
    """Two-dimensional depthwise convolution.

    Uses SAME padding with 0s, a stride of 1 and no dilation. A single output
    channel is used per input channel (channel_multiplier=1).

    input: input array with shape (height, width, in_depth)
    w: filter array with shape (fd, fd, in_depth)

    Returns a result with shape (height, width, in_depth).
    """
    height, width, in_depth = input.shape
    output_height = height - w.shape[0] + 1
    output_width = width - w.shape[1] + 1
    output = np.zeros((output_height, output_width, in_depth))
    for c in range(in_depth):
        # For each input channel separately, apply its corresponsing filter
        # to the input.
        for i in range(output_height):
            for j in range(output_width):
                for fi in range(w.shape[0]):
                    for fj in range(w.shape[1]):
                        w_element = w[fi, fj, c]
                        output[i, j, c] += (
                            input[i + fi, j + fj, c] * w_element)
    return output


# Define the input tensor, kernel, and parameters
input_tensor = np.random.rand(6, 6, 3).astype(np.float32)
kernel = np.random.rand(3, 3, 3).astype(np.float32)
input_height, input_width, input_channels = input_tensor.shape
kernel_size = kernel.shape[0]

# Calculate the output tensor shape
output_height = input_height - kernel_size + 1
output_width = input_width - kernel_size + 1

# Create an empty output tensor
output_ctypes = np.zeros((output_height, output_width, input_channels), dtype=np.float32)

# Convert the arrays to contiguous memory for ctypes
input_ptr = input_tensor.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
kernel_ptr = kernel.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
output_ptr = output_ctypes.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
# Calculate the result using numpy for comparison
output_np = depthwise_conv2d(input_tensor, kernel).astype("float32")

    
file_name = "depthwiseconv.cpp"
so_name = "depthwiseconv.so"
# Load the shared library with the batch matrix multiplication function
success, output = run_compilation(so_name, file_name)
# # os.remove(file_name)
lib = ctypes.CDLL(os.path.join(os.getcwd(), so_name))
function = getattr(lib, "depthwiseconv_kernel")
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