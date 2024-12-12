import argparse
import ctypes
import math
import os
import subprocess
from ctypes import CDLL

import numpy as np
import torch
import torch.nn.functional as F

from benchmark.template.mlu_host_template import create_bang_func
from benchmark.utils import run_mlu_compilation as run_compilation


def ref_program(q, k, v, causal=False):
    score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    if causal:
        mask = torch.triu(torch.ones(score.shape[-2], score.shape[-1]), diagonal=1)
        mask = mask.masked_fill(mask == 1, torch.finfo(q.dtype).min)
        mask = mask.to(q.device, q.dtype)
        score = score + mask
    attn = F.softmax(score, dim=-1)
    output = torch.matmul(attn, v)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="the source file")
    args = parser.parse_args()
    base_name = os.path.basename(args.file)
    name = "mha"
    causal = False
    shapes = base_name.split(".")[0]
    shape = [int(intg) for intg in shapes.split("_")[1:]]
    dtype = torch.float32

    query = torch.randn(shape).to(dtype).contiguous()
    key = torch.randn(shape).to(dtype).contiguous()
    value = torch.randn(shape).to(dtype).contiguous()
    file_name = create_bang_func(args.file)
    so_name = args.file.replace(".mlu", ".so")
    success, output = run_compilation(so_name, file_name)
    os.remove(file_name)
    lib = CDLL(os.path.join(os.getcwd(), so_name))
    # 获取函数句柄
    function = getattr(lib, name + "_kernel")
    # 定义函数参数和返回类型
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]
    function.restype = None
    # 创建输入数组
    expected_output = ref_program(query, key, value)
    # 创建输出数组
    output_array = np.zeros_like(query.numpy())
    # 将输入数组和输出数组转换为C指针类型
    input_ptr_q = query.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    input_ptr_k = key.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    input_ptr_v = value.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_ptr = output_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # 调用CUDA kernel
    function(input_ptr_q, input_ptr_k, input_ptr_v, output_ptr, np.prod(shape))
    # 验证结果

    # 验证结果
    np.testing.assert_allclose(
        output_array,
        expected_output.numpy(),
        rtol=1e-03,
        atol=1e-03,
        equal_nan=True,
        err_msg="",
        verbose=True,
    )

    print("验证通过！")
    result = subprocess.run(["rm", so_name])
