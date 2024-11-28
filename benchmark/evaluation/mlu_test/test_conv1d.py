import argparse
import os

import numpy as np
import toc
import tvm
import tvm.topi.testing
from toc import Environment
from tvm import te

env = Environment("cambricon/mlu590-h8")


def verify_conv1d(name, file, shape, kernel_shape, output_shape):
    op_name = name.split("_")[0]
    A = te.placeholder(shape, dtype="float32", name="A")
    B = te.placeholder(kernel_shape, dtype="float32", name="B")

    A_buff = tvm.tir.decl_buffer(A.shape, "float32", "A_buf")
    B_buff = tvm.tir.decl_buffer(B.shape, "float32", "B_buf")
    C_buff = tvm.tir.decl_buffer(output_shape, "float32", "C_buf")

    @tvm.register_func("toc_callback_bang_postproc")
    def toc_callback_bang_postproc(code):
        with open(file, "r") as f:
            code = f.read()
            f.close()
        code = code.replace("void " + op_name + "(", "void " + op_name + "_kernel0(")
        return code

    def test_conv1d(A, B, C):
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
        return body

    C = te.extern(
        output_shape,
        [A, B],
        lambda ins, outs: test_conv1d(ins[0], ins[1], outs[0]),
        name=op_name,
        dtype="float32",
    )

    s = te.create_schedule(C.op)

    dev = tvm.device("bang", 0)
    input_array = np.random.rand(*shape).astype("float32")
    kernel = np.random.rand(*kernel_shape).astype("float32")
    a = tvm.nd.array(input_array, dev)
    b = tvm.nd.array(kernel, dev)
    c = tvm.nd.array(np.random.rand(*output_shape).astype("float32"), dev)
    with toc.build_config(env):
        f = toc.build(s, [A, B, C], name=op_name)
    f(a, b, c)
    tvm._ffi.registry.remove_global_func("toc_callback_bang_postproc")

    # Calculate the result using numpy for comparison
    output_np = np.convolve(input_array, kernel, mode="valid")
    # Check if the results match
    np.testing.assert_allclose(
        c.numpy(),
        output_np,
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
    shape = [shape[1]]
    kernel_shape = [3]
    output_shape = [shape[0]]
    verify_conv1d(base_name, args.file, shape, kernel_shape, output_shape)
    print("验证通过！")
