import argparse
import os

import numpy as np
import toc
import tvm
import tvm.topi.testing
from toc import Environment
from tvm import te

env = Environment("cambricon/mlu590-h8")


# Define the batch matrix multiplication function using numpy
def batch_matmul(A, B):
    return np.matmul(A, B)


def verify_bmm(name, file, shape_A, shape_B, shape_C):

    A = te.placeholder(shape_A, name="A", dtype="float32")
    B = te.placeholder(shape_B, name="B", dtype="float32")
    op_name = name.split("_")[0]

    @tvm.register_func("toc_callback_bang_postproc")
    def toc_callback_bang_postproc(code):
        with open(file, "r") as f:
            code = f.read()
            f.close()
        code = code.replace("void " + op_name + "(", "void " + op_name + "_kernel0(")
        return code

    def bmm(A, B, C):
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
        with ib.for_range(0, n, name="i") as i:
            Cptr[i] = Aptr[i] + Bptr[i]
        body = ib.get()
        return body  # cpu compute

    C = te.extern(
        shape_C,
        [A, B],
        lambda ins, outs: bmm(ins[0], ins[1], outs[0]),
        name="bmm",
        dtype="float32",
    )
    dev = tvm.device("bang", 0)
    s = te.create_schedule(C.op)
    a = np.ones(shape_A).astype("float32")
    b = np.ones(shape_B).astype("float32")
    # Perform batch matrix multiplication using numpy
    result_np = batch_matmul(a, b)

    a = tvm.nd.array(a, dev)
    b = tvm.nd.array(b, dev)
    c = tvm.nd.array(np.random.rand(*shape_C).astype("float32"), dev)
    with toc.build_config(env):
        f = toc.build(s, [A, B, C], name=op_name)
    f(a, b, c)

    np.testing.assert_allclose(
        c.numpy(),
        result_np,
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
    batch_size, matrix_dim_i, matrix_dim_j, matrix_dim_k = shape
    shape_A = [batch_size, matrix_dim_i, matrix_dim_j]
    shape_B = [batch_size, matrix_dim_j, matrix_dim_k]
    shape_C = [batch_size, matrix_dim_i, matrix_dim_k]
    verify_bmm(base_name, args.file, shape_A, shape_B, shape_C)
    print("验证通过！")
