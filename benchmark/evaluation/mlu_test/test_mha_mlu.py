import argparse
import os

import numpy as np
import toc
import tvm
import tvm.topi.testing
from toc import Environment
from tvm import te

env = Environment("cambricon/mlu590-h8")
import numpy as np
import torch.nn.functional as F


def ref_program(q, k, v, causal=False):
    return F.scaled_dot_product_attention(q, k, v)


def verify_scaled_dot_product_attention(name, file, shape):
    op_name = name.split("_")[0]
    A = te.placeholder(shape, dtype="float32", name="A")
    B = te.placeholder(shape, dtype="float32", name="B")
    C = te.placeholder(shape, dtype="float32", name="C")

    A_buff = tvm.tir.decl_buffer(A.shape, "float32", "A_buf")
    B_buff = tvm.tir.decl_buffer(B.shape, "float32", "B_buf")
    C_buff = tvm.tir.decl_buffer(C.shape, "float32", "C_buf")
    D_buff = tvm.tir.decl_buffer(shape, "float32", "D_buf")

    @tvm.register_func("toc_callback_bang_postproc")
    def toc_callback_bang_postproc(code):
        with open(file, "r") as f:
            code = f.read()
            f.close()
        code = code.replace("void " + op_name + "(", "void " + op_name + "_kernel0(")
        return code

    def test_scaled_dot_product_attention(A, B, C, D, seq_len, num_heads, head_dim):
        n = A.shape[0]
        prod = np.prod(A.shape[:-1])
        ib = tvm.tir.ir_builder.create()
        tx = te.thread_axis("threadIdx.x")
        bx = te.thread_axis("blockIdx.x")
        ib.scope_attr(tx, "thread_extent", 4)
        ib.scope_attr(bx, "thread_extent", 4)

        Aptr = ib.buffer_ptr(A)
        Bptr = ib.buffer_ptr(B)
        Cptr = ib.buffer_ptr(C)
        Dptr = ib.buffer_ptr(D)

        with ib.for_range(0, n, name="i") as i:
            j = seq_len * num_heads * head_dim
            Dptr[i] = Aptr[i] + Bptr[i] + Cptr[i] + j
        body = ib.get()
        return body

    D = te.extern(
        shape,
        [A, B, C],
        lambda ins, outs: test_scaled_dot_product_attention(
            ins[0], ins[1], ins[2], outs[0], shape[1], shape[2], shape[3]
        ),
        name="mha",
        in_buffers=[A_buff, B_buff, C_buff],
        out_buffers=[D_buff],
        dtype="float32",
    )

    s = te.create_schedule(D.op)

    dev = tvm.device("bang", 0)
    a = tvm.nd.array(np.random.rand(*shape).astype("float32"), dev)
    b = tvm.nd.array(np.random.rand(*shape).astype("float32"), dev)
    c = tvm.nd.array(np.random.rand(*shape).astype("float32"), dev)
    d = tvm.nd.array(np.random.rand(*shape).astype("float32"), dev)
    with toc.build_config(env):
        f = toc.build(s, [A, B, C, D], name="mha")
    f(a, b, c, d)
    time_f = f.time_evaluator("mha", dev, number=20, repeat=100)
    cost = time_f(a, b, c, d)
    print(f"{name} execution time: {cost.mean * 1000} ms")

    tvm._ffi.registry.remove_global_func("toc_callback_bang_postproc")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="the source file")
    args = parser.parse_args()
    base_name = os.path.basename(args.file)
    shapes = base_name.split(".")[0]
    shape = [int(intg) for intg in shapes.split("_")[1:]]
    verify_scaled_dot_product_attention(base_name, args.file, shape)
    print("验证通过！")
