import numpy as np
import ctypes
import subprocess
import os
import argparse

def run_compilation(so_name, file_name):
    try:
        output = subprocess.run(
            ["nvcc", "-shared", "-Xcompiler", "-fPIC", "-o", so_name, file_name],
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

def get_im2col_indices(images_shape, filter_shape, padding, stride):
    """Get index for shape"""
    batch_size, channels, height, width = images_shape
    filter_height, filter_width = filter_shape
    pad_h, pad_w = padding
    stride_h, stride_w = stride
    out_height = int((height + np.sum(pad_h) - filter_height) / stride_h + 1)
    out_width = int((width + np.sum(pad_w) - filter_width) / stride_w + 1)

    i0 = np.repeat(np.arange(filter_height), filter_width)
    i0 = np.tile(i0, channels)
    i1 = stride_h * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(filter_width), filter_height * channels)
    j1 = stride_w * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(channels), filter_height * filter_width).reshape(-1, 1)
    return (k, i, j)


def image_to_column(images, filter_shape, stride, pad):
    """Transpose the input for conv"""
    filter_height, filter_width = filter_shape
    pad_h, pad_w = cpu_pad(pad)
    images_padded = np.pad(images, ((0, 0), (0, 0), pad_h, pad_w), mode="constant")
    k, i, j = get_im2col_indices(images.shape, filter_shape, (pad_h, pad_w), stride)
    cols = images_padded[:, k, i, j]
    channels = images.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(filter_height * filter_width * channels, -1)
    return cols


def cpu_pad(padding=None):
    """Get the pad param"""
    pad_h = 0
    pad_w = 0
    if isinstance(padding, (list, tuple)):
        if len(padding) == 2:
            pad_h = padding[0]
            pad_w = padding[1]
        elif len(padding) == 1:
            pad_h = pad_w = padding[0]
    elif isinstance(padding, int):
        pad_h = pad_w = padding
    return (pad_h, pad_h), (pad_w, pad_w)


def cpu_conv(data, kernel, stride_w, stride_h, pad=None):
    """Conv op in cpu"""
    batch_size, input_height, input_width, input_channel = data.shape
    output_channel, kernel_height, kernel_width, _ = kernel.shape
    pad_h, pad_w = cpu_pad(pad)
    out_height = int((input_height + np.sum(pad_h) - kernel_height) / stride_h + 1)
    out_width = int((input_width + np.sum(pad_w) - kernel_width) / stride_w + 1)
    data = data.transpose(0, 3, 1, 2)
    kernel = kernel.transpose(0, 3, 1, 2)
    X_col = image_to_column(
        data, (kernel_height, kernel_width), (stride_h, stride_w), pad
    )
    W_col = kernel.reshape((output_channel, -1))
    output = W_col.dot(X_col)
    output = output.reshape(output_channel, out_height, out_width, batch_size)
    return output.transpose(3, 1, 2, 0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="the source file")

    args = parser.parse_args()
    base_name = os.path.basename(args.file)

    name = base_name.split("_")[0]
    data_shape = base_name.split("_")[1:5]

    data_shape = [int(intg) for intg in data_shape]

    kernel_shape = base_name.split("_")[5:9]
    kernel_shape = [int(intg) for intg in kernel_shape]
    stride_h = stride_w = int(base_name.split("_")[9])
    pad = int(base_name.split("_")[10])
    dtype = base_name.split("_")[-2].replace(".cu", "")
    wtype = base_name.split("_")[-1].replace(".cu", "")

    # generate data
    data_np = np.random.uniform(low=1.0, high=2.0, size=data_shape).astype(dtype)
    kernel_np = np.random.uniform(low=1.0, high=2.0, size=kernel_shape).astype(dtype)
    # cpu compute
    result_cpu = cpu_conv(data_np, kernel_np, stride_h, stride_w, pad)  
    
    # Load the shared library with the conv2d function
    so_name = args.file.replace(".cu", ".so")
    with open(args.file, "r") as f:
        code = f.read()
        f.close()

    with open("./macro/cuda_macro.txt", "r") as f:
        macro = f.read()
        f.close()
    code = macro + code

    file_name = args.file.replace(base_name.replace(".cu", ""), base_name + "_bak.cu")
    with open(file_name, mode="w") as f:
        f.write(code)
        f.close()
    success, output = run_compilation(so_name, file_name)
    os.remove(file_name)
    # # os.remove(file_name)
    lib = ctypes.CDLL(os.path.join(os.getcwd(), so_name))
    function = getattr(lib, "conv1d")
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
    result = subprocess.run(["rm", so_name])