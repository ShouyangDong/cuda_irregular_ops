import argparse
import os

import numpy as np
import toc
import torch
import tvm
import tvm.topi.testing
from toc import Environment
from tvm import te
from tvm.topi.utils import get_const_tuple

env = Environment("cambricon/mlu590-h8")


def ref_program(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)


def perf_rmsnorm(name, file, shape):
    op_name = name.split("_")[0]
    A = te.placeholder(shape, dtype="float32", name="A")

    @tvm.register_func("toc_callback_bang_postproc")
    def toc_callback_bang_postproc(code):
        with open(file, "r") as f:
            code = f.read()
            f.close()
        code = code.replace("void " + op_name + "(", "void " + op_name + "_kernel0(")
        return code

    def test_rmsnorm(A, B):
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
        lambda ins, outs: test_rmsnorm(ins[0], outs[0]),
        name=op_name,
        dtype="float32",
    )

    s = te.create_schedule(B.op)

    dev = tvm.device("bang", 0)
    input_array = np.random.uniform(size=shape).astype("float32")
    expected_output = ref_program(torch.from_numpy(input_array))

    a = tvm.nd.array(input_array, dev)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), dev)
    with toc.build_config(env):
        f = toc.build(s, [A, B], name=op_name)
    f(a, b)
    tvm._ffi.registry.remove_global_func("toc_callback_bang_postproc")
    np.testing.assert_allclose(
        b.numpy(),
        expected_output,
        rtol=1e-03,
        atol=1e-03,
        equal_nan=True,
        err_msg="",
        verbose=True,
    )


if __name__ == "__main__":
    name = "rms_norm"
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="the source file")
    args = parser.parse_args()
    base_name = os.path.basename(args.file)
    shapes = base_name.split(".")[0]
    shape = [int(intg) for intg in shapes.split("_")[1:]]
    perf_rmsnorm(base_name, args.file, shape)
    print("验证通过！")
