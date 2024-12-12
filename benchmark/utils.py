import subprocess

import numpy as np


def avgpool_np(data, kernel_stride):
    """avg pooling with numpy
    data : numpy.array
        input array

    kernel : list or tuple
        The kernel of avgpool

    stride : list or tuple
        The stride of avgpool
    """
    batch, dh, dw, dc = data.shape
    kh, kw, sh, sw = kernel_stride
    ch = (dh - kh) // sh + 1
    cw = (dw - kw) // sw + 1
    ret = np.zeros((batch, ch, cw, dc))
    for i in range(ch):
        for j in range(cw):
            mask = data[:, i * sh : i * sh + kh, j * sw : j * sw + kw, :]
            ret[:, i, j, :] = np.average(mask, axis=(1, 2))
    return ret


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


def maxpool_np(data, kernel_stride):
    """max pooling with numpy
    data : numpy.array
        input array

    kernel : list or tuple
        The kernel of avgpool

    stride : list or tuple
        The stride of avgpool
    """
    batch, dh, dw, dc = data.shape
    kh, kw, sh, sw = kernel_stride
    ch = (dh - kh) // sh + 1
    cw = (dw - kw) // sw + 1
    ret = np.zeros((batch, ch, cw, dc))
    for i in range(ch):
        for j in range(cw):
            mask = data[:, i * sh : i * sh + kh, j * sw : j * sw + kw, :]
            ret[:, i, j, :] = np.max(mask, axis=(1, 2))
    return ret


def minpool_np(data, kernel_stride):
    """min pooling with numpy
    data : numpy.array
        input array

    kernel : list or tuple
        The kernel of avgpool

    stride : list or tuple
        The stride of avgpool
    """
    batch, dh, dw, dc = data.shape
    kh, kw, sh, sw = kernel_stride
    ch = (dh - kh) // sh + 1
    cw = (dw - kw) // sw + 1
    ret = np.zeros((batch, ch, cw, dc))
    for i in range(ch):
        for j in range(cw):
            mask = data[:, i * sh : i * sh + kh, j * sw : j * sw + kw, :]
            ret[:, i, j, :] = np.min(mask, axis=(1, 2))
    return ret


def conv2d_nchw(input_tensor, kernel, stride, pad=0):
    # 获取输入张量和卷积核的维度
    N, C, H, W = input_tensor.shape
    out_channels, in_channels, kernel_height, kernel_width = kernel.shape
    # 计算输出张量的空间维度
    out_height = (H - kernel_height) // stride + 1
    out_width = (W - kernel_width) // stride + 1

    # 初始化输出张量
    output_tensor = np.zeros((N, out_channels, out_height, out_width))

    # 执行卷积操作
    for n in range(N):
        for out_channel in range(out_channels):
            for in_channel in range(in_channels):
                for i in range(0, out_height):
                    for j in range(0, out_width):
                        h_start = i * stride
                        h_end = h_start + kernel_height
                        w_start = j * stride
                        w_end = w_start + kernel_width
                        output_tensor[n, out_channel, i, j] += np.sum(
                            input_tensor[n, in_channel, h_start:h_end, w_start:w_end]
                            * kernel[out_channel, in_channel]
                        )

    return output_tensor


def run_dlboost_compilation(so_name, file_name):
    try:
        output = subprocess.run(
            [
                "g++",
                "-shared",
                "-fPIC",
                "-march=icelake-server",
                "-O3",
                file_name,
                "-o",
                so_name,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            check=True,
            timeout=15,
        )
        return True, output
    except subprocess.CalledProcessError as e:
        return False, e.output


def run_cpp_compilation(so_name, file_name):
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


def run_mlu_compilation(so_name, file_name):
    try:
        output = subprocess.run(
            [
                "cncc",
                "-shared",
                "-fPIC",
                "--bang-mlu-arch=mtp_592",
                "-o",
                so_name,
                file_name,
            ],
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


def run_cuda_compilation(so_name, file_name):
    try:
        output = subprocess.run(
            [
                "nvcc",
                "-Xcompiler",
                "-fPIC",
                "-shared",
                "-arch=sm_80",
                "-o",
                so_name,
                file_name,
            ],
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


def run_test(file_name, test_file):
    try:
        output = subprocess.run(
            ["python", test_file, "--file", file_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            check=True,
            text=True,
            timeout=400,
        )
        return True, output
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except subprocess.CalledProcessError as e:
        return False, e.output
