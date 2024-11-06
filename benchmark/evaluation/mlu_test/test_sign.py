import argparse
import os

import bangpy
import numpy as np
import toc
import tvm
import tvm.topi.testing
from toc import Environment
from tvm import te
from tvm.topi.utils import get_const_tuple

env = Environment("cambricon/mlu590-h8")


def verify_sign(name, file, shape):
    from toc import Environment

    env = Environment("cambricon/mlu590-h8")

    @tvm.register_func("toc_callback_bang_postproc")
    def toc_callback_bang_postproc(code):
        with open(file, "r") as f:
            code = f.read()
            f.close()
        code = code.replace("void " + op_name + "(", "void " + op_name + "_kernel0(")
        return code

    op_name = name.split("_")[0]
    A = te.placeholder(shape, dtype="float32", name="A")

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
    bangpy.assert_allclose(b.numpy(), np.sign(data))
    tvm._ffi.registry.remove_global_func("toc_callback_bang_postproc")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="the source file")
    args = parser.parse_args()
    base_name = os.path.basename(args.file)
    shapes = base_name.split(".")[0]
    shape = [int(intg) for intg in shapes.split("_")[1:]]
    verify_sign(base_name, args.file, shape)
    print("验证通过！")
