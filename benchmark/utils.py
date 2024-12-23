import subprocess

import numpy as np
import torch
import torch.nn.functional as F


def avgpool_np(input_tensor, kernel_stride):
    input_tensor = input_tensor.permute(0, 3, 1, 2)
    avgpool = torch.nn.AvgPool2d(
        kernel_size=kernel_stride[:2], stride=kernel_stride[2:]
    )
    # 执行平均池化
    output_tensor = avgpool(input_tensor)
    output_tensor = output_tensor.permute(0, 2, 3, 1)
    return output_tensor


def sumpool_np(input_tensor, kernel_stride):
    input_tensor = input_tensor.permute(0, 3, 1, 2)
    avgpool = torch.nn.AvgPool2d(
        kernel_size=kernel_stride[:2], stride=kernel_stride[2:]
    )
    # 执行平均池化
    output_tensor = avgpool(input_tensor)
    output_tensor = output_tensor.permute(0, 2, 3, 1)
    return output_tensor * kernel_stride[0] * kernel_stride[1]


def maxpool_np(input_tensor, kernel_stride):
    input_tensor = input_tensor.permute(0, 3, 1, 2)
    avgpool = torch.nn.AvgPool2d(
        kernel_size=kernel_stride[:2], stride=kernel_stride[2:]
    )
    # 执行平均池化
    output_tensor = avgpool(input_tensor)
    output_tensor = output_tensor.permute(0, 2, 3, 1)
    return output_tensor


def minpool_np(input_tensor, kernel_stride):
    class MinPool2d(torch.nn.Module):
        def __init__(self, kernel_size, stride=None, padding=0):
            super(MinPool2d, self).__init__()
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding

        def forward(self, x):
            # 取反输入
            x_neg = -x
            # 执行最大池化
            x_maxpool = F.max_pool2d(
                x_neg, self.kernel_size, stride=self.stride, padding=self.padding
            )
            # 再取反结果
            return -x_maxpool

    # 使用自定义的 MinPool2d
    pool = MinPool2d(kernel_size=kernel_stride[:2], stride=kernel_stride[2:])
    input_tensor = input_tensor.permute(0, 3, 1, 2)
    output_tensor = pool(input_tensor)
    output_tensor = output_tensor.permute(0, 2, 3, 1)
    return output_tensor


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
        print(e.output)
        return False, e.output
