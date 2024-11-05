import argparse
import ctypes
import os
import subprocess

import os

import bangpy
import numpy as np
import toc
import torch
import tvm
import tvm.topi.testing
from bangpy import tensor_op as tsop
from toc import compile_bang
from tvm import te, topi
from tvm.topi.utils import get_const_tuple

from toc import Environment
env = Environment("cambricon/mlu590-h8")

def ref_program(x):
    # 对最后一个维度进行softmax操作
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def verify_softmax(name, file, shape):
    op_name = name.split("_")[0]
    A = te.placeholder(shape, dtype="float32", name="A")
    @tvm.register_func
    def toc_callback_bang_postproc(code, file):
        tvm._ffi.registry.remove_global_func("toc_callback_bang_postproc")
        if not os.path.exists(file):
            with open(file, "w", encoding="utf-8") as f:
                f.write(code)
        return code

    def test_activation(A, B):
        n = A.shape[0]
        prod = np.prod(A.shape[:-1])
        ib = tvm.tir.ir_builder.create()
        tx = te.thread_axis("threadIdx.x")
        bx = te.thread_axis("blockIdx.x")

        ib.scope_attr(tx, "thread_extent", 4)
        ib.scope_attr(bx, "thread_extent", 4)

        Aptr = ib.buffer_ptr(A)
        Bptr = ib.buffer_ptr(B)
        with ib.for_range(0, n, name="i") as i:
            Bptr[i] = Aptr[i]
        body = ib.get()
        return body

    B = te.extern(
        A.shape,
        [A],
        lambda ins, outs: test_activation(ins[0], outs[0]),
        name=op_name,
        dtype="float32",
    )
    s = te.create_schedule(B.op)

    dev = tvm.device("bang", 0)
    data = np.random.rand(*shape).astype("float32")
    a = tvm.nd.array(data, dev)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), dev)

    with toc.build_config(env):
        f = toc.build(s, [A, B], name=op_name)
    f(a, b)
    cpu_result = ref_program(data)
    # Check if the results match
    np.testing.assert_allclose(
        b.numpy(),
        cpu_result,
        rtol=1e-03,
        atol=1e-03,
        equal_nan=True,
        err_msg="",
        verbose=True,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="the source file")
    args = parser.parse_args()
    base_name = os.path.basename(args.file)
    shapes = base_name.split(".")[0]
    shape = [int(intg) for intg in shapes.split("_")[1:]]
    verify_softmax(base_name, args.file, shape)
    print("验证通过！")
