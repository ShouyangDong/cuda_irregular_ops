import numpy as np
from ctypes import CDLL, c_void_p, c_double, c_int
import subprocess
import glob
import os
import math
import ctypes
import argparse

from multi_head_attention import MultiHeadAttention

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

def ref_program(q, k, v, causal=False):
    import torch
    import torch.nn.functional as F
    score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    if causal:
        mask = torch.triu(torch.ones(score.shape[-2], score.shape[-1]), diagonal=1)
        mask = mask.masked_fill(mask==1, torch.finfo(q.dtype).min)
        mask = mask.to(q.device, q.dtype)
        score = score + mask
    attn = F.softmax(score, dim=-1)
    output = torch.matmul(attn, v)
    return o


if __name__ == "__main__":
    name = "multiHeadAttentionForward"
    causal = False
    shape = [64, 2048, 12, 256]
    dtype = "float32"
    query = torch.randn(shape, dtype)
    key = torch.randn(shape, dtype)
    value = torch.randn(shape, dtype)

    file_name = "mha.cpp"
    so_name = "mha.so"
    
    success, output = run_compilation(so_name, file_name)
    lib = CDLL(os.path.join(os.getcwd(), so_name))
    function = getattr(lib, name + "_kernel")
    # 定义函数参数和返回类型
    function.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float)
    ]
    function.restype = None
    # 创建输入数组
    dtype = "float32"
    attn = MultiHeadAttention(E, num_heads)
    attn_value, attn_weights = attn(query, key, value, attn_mask)

    # 创建输出数组
    output_array = np.zeros_like(input_array)

    # 将输入数组和输出数组转换为C指针类型
    input_ptr = input_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_ptr = output_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    # 调用C函数
    function(input_ptr, output_ptr)
    # 验证结果

    np.testing.assert_allclose(
        output_array,
        expected_output,
        rtol=1e-03,
        atol=1e-03,
        equal_nan=True,
        err_msg="",
        verbose=True,
    )

    print("验证通过！")
    import time 
    t1 = time.time()
    for i in range(20):
        function(input_ptr, output_ptr)    
    t2 = time.time()
    cost = (t2 - t1) / 20.0 * 1e3
    print("[INFO]*******cost: ", cost)
