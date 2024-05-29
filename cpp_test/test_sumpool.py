import numpy as np
import ctypes
import subprocess
import os
import argparse

def run_compilation(so_name, file_name):
    try:
        output = subprocess.run(
            ["g++", "-shared", "-fPIC", "-O3", file_name, "-o", so_name],
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


def sumpool_np(data, kernel_stride):
    """sum pooling with numpy
    data : numpy.array
        input array

    kernel : list or tuple
        The kernel of sumpool

    stride : list or tuple
        The stride of sumpool
    """
    batch, dh, dw, dc = data.shape
    kh, kw, sh, sw = kernel_stride
    ch = (dh - kh) // sh + 1
    cw = (dw - kw) // sw + 1
    ret = np.zeros((batch, ch, cw, dc))
    for i in range(ch):
        for j in range(cw):
            mask = data[:, i * sh : i * sh + kh, j * sw : j * sw + kw, :]
            ret[:, i, j, :] = np.sum(mask, axis=(1, 2))
    return ret

def generate_data(shape, dtype):
    return np.random.uniform(size=shape).astype(dtype)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="the source file")
    args = parser.parse_args()
    base_name = os.path.basename(args.file)

    name = base_name.split(".")[0].split("_")[0]
    shape = base_name.split(".")[0].split("_")[1:5]
    shape = [int(intg) for intg in shape]
    kernel_stride = base_name.split(".")[0].split("_")[5:]
    kernel_stride = [int(intg) for intg in kernel_stride]

    dtype = "float32"

    input_array = generate_data(shape, dtype)
    # Calculate the result using numpy for comparison
    output_np = sumpool_np(input_array, kernel_stride)
    output_array = np.zeros(shape=output_np.shape, dtype=dtype)

    # Convert the arrays to contiguous memory for ctypes
    input_ptr = input_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_ptr = output_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Load the shared library with the avgpool function
    so_name = args.file.replace(".cpp", ".so")
    with open(args.file, "r") as f:
        code = f.read()
        f.close()

    with open("./macro/cpp_macro.txt", "r") as f:
        macro = f.read()
        f.close()
    code = macro + code

    file_name = args.file.replace(base_name.replace(".cpp", ""), base_name + "_bak.cpp")
    with open(file_name, mode="w") as f:
        f.write(code)
        f.close()
    success, output = run_compilation(so_name, file_name)
    os.remove(file_name)

    lib = ctypes.CDLL(os.path.join(os.getcwd(), so_name))
    function = getattr(lib, name + "_kernel")
    # 定义函数参数和返回类型
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    function.restype = None
    # Call the function with the matrices and dimensions
    function(output_ptr, input_ptr)
    # Check if the results match
    np.testing.assert_allclose(
        output_array,
        output_np,
        rtol=1e-03,
        atol=1e-03,
        equal_nan=True,
        err_msg="",
        verbose=True,
    )
    print("验证通过！")
    result = subprocess.run(["rm", so_name])
