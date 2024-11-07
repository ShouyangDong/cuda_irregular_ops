import argparse
import os

import numpy as np
import toc
import tvm
import tvm.topi.testing
from toc import Environment
from tvm import te

env = Environment("cambricon/mlu590-h8")


def ref_program(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    x_normalized = (x - mean) / (std + eps)
    out = gamma * x_normalized + beta
    return out


def verify_layernorm(name, file, shape):
    op_name = name.split("_")[0]
    A = te.placeholder(shape, dtype="float32", name="A")
    B = te.placeholder(shape[-1:], dtype="float32", name="B")
    C = te.placeholder(shape[-1:], dtype="float32", name="C")

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

    def test_layernorm(A, B, C, D):
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
            Dptr[i] = Aptr[i] + Bptr[i] + Cptr[i]
        body = ib.get()
        return body

    D = te.extern(
        A.shape,
        [A, B, C],
        lambda ins, outs: test_layernorm(ins[0], ins[1], ins[2], outs[0]),
        name=op_name,
        dtype="float32",
    )

    s = te.create_schedule(D.op)

    dev = tvm.device("bang", 0)
    input_array = np.random.uniform(size=shape).astype("float32")
    gamma_array = np.random.uniform(size=shape[-1:]).astype("float32")
    beta_array = np.random.uniform(size=shape[-1:]).astype("float32")
    expected_output = ref_program(input_array, gamma_array, beta_array)

    a = tvm.nd.array(input_array, dev)
    b = tvm.nd.array(gamma_array, dev)
    c = tvm.nd.array(beta_array, dev)
    d = tvm.nd.array(np.random.rand(*shape).astype("float32"), dev)
    with toc.build_config(env):
        f = toc.build(s, [A, B, C, D], name=op_name)
    f(a, b, c, d)

    np.testing.assert_allclose(
        d.numpy(),
        expected_output,
        rtol=1e-03,
        atol=1e-03,
        equal_nan=True,
        err_msg="",
        verbose=True,
    )
    tvm._ffi.registry.remove_global_func("toc_callback_bang_postproc")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="the source file")
    args = parser.parse_args()
    base_name = os.path.basename(args.file)
    shapes = base_name.split(".")[0]
    shape = [int(intg) for intg in shapes.split("_")[1:]]
    verify_layernorm(base_name, args.file, shape)
    print("验证通过！")
