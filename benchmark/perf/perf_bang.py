import glob
import os

import bangpy
import numpy as np
import toc
import torch
import tvm
import tvm.topi.testing
from bangpy import tensor_op as tsop
from toc import Environment, compile_bang
from tvm import te
from tvm.topi.utils import get_const_tuple

from benchmark.utils import avgpool_np, maxpool_np, minpool_np, sumpool_np

env = Environment("cambricon/mlu590-h8")


@tvm.register_func("toc_callback_bang_compile", override=True)
def toc_callback_bang_compile(code):
    cnfatbin = compile_bang(
        code,
        "mtp_592",
        cncc_args=None,
        cnas_args=["-fno-soft-pipeline", "-fno-if-conversion"],
        opt_level="O3",
        output_file=None,
    )
    return cnfatbin


def perf_elementwise(name, file, shape):
    op_name = name.split("_")[0]

    @tvm.register_func("toc_callback_bang_postproc", override=True)
    def toc_callback_bang_postproc(code):
        tvm._ffi.registry.remove_global_func("toc_callback_bang_postproc")
        if not os.path.exists(file):
            with open(file, "w", encoding="utf-8") as f:
                f.write(code)
        code = open(file, encoding="utf-8").read()

        code = code.replace(
            "void " + op_name + "(", "void " + op_name + "_kernel0("
        )

        return code

    if op_name == "add":
        input0 = tsop.tensor(shape, dtype=bangpy.float32, name="input0")
        input1 = tsop.tensor(shape, dtype=bangpy.float32, name="input1")
        # Describe Computation
        result = tsop.add(input0, input1)
        # Build and get executable module
        fmlu = tsop.BuildBANG(
            [input0, input1], [result], "mlu590-h8", kernel_name=op_name
        )
        # Generate random test data and run on mlu and cpu
        data_lhs = np.random.rand(*shape).astype("float32")
        data_rhs = np.random.rand(*shape).astype("float32")
        result_np = np.zeros(shape=shape, dtype="float32")
        dev = bangpy.device(0)
        data_lhs_dev = bangpy.Array(data_lhs, dev)
        data_rhs_dev = bangpy.Array(data_rhs, dev)
        result_arr = bangpy.Array(result_np, dev)

        fmlu(data_lhs_dev, data_rhs_dev, result_arr)
        # cpu_output = np.add(data_lhs, data_rhs)
        # bangpy.assert_allclose(mlu_output.numpy(), cpu_output)
        # print("验证通过！")
        evaluator = fmlu.time_evaluator(number=100, repeat=1, min_repeat_ms=0)
        cost = evaluator(data_lhs_dev, data_rhs_dev, result_arr).mean
        print(f"{name} execution time: {cost * 1000} ms")
    elif op_name == "sign":
        input0 = tsop.tensor(shape, dtype=bangpy.float32, name="input0")
        # Describe Computation
        result = tsop.sign(input0)
        # Build and get executable module
        fmlu = tsop.BuildBANG(
            [input0], [result], "mlu590-h8", kernel_name=op_name
        )
        # Generate random test data and run on mlu and cpu
        data_npy = np.random.rand(*shape).astype("float32")
        out_npy = np.sign(data_npy)
        result_np = np.random.uniform(size=shape).astype("float32")
        dev = bangpy.device(0)
        data_dev = bangpy.Array(data_npy, dev)
        result_arr = bangpy.Array(result_np, dev)

        fmlu(data_dev, result_arr)

        np.testing.assert_allclose(
            result_arr.numpy(),
            out_npy,
            equal_nan=True,
            err_msg="",
            verbose=True,
        )
        print("验证通过！")
        evaluator = fmlu.time_evaluator(number=100, repeat=1, min_repeat_ms=0)
        cost = evaluator(data_dev, result_arr).mean * 1e3
        print(f"{name} execution time: {cost} ms")
        func_name = "toc_callback_bang_postproc"
        tvm._ffi.registry.remove_global_func(func_name)


def perf_pooling(name, file, shape, kernel, stride):
    op_name = name.split("_")[0]

    kh, kw = kernel[0], kernel[1]
    sh, sw = stride[0], stride[1]

    _op2np = {
        "avgpool": avgpool_np,
        "sumpool": sumpool_np,
        "maxpool": maxpool_np,
        "minpool": minpool_np,
    }

    _opTensorOp = {
        "avgpool": tsop.avgpool,
        "sumpool": tsop.sumpool,
        "maxpool": tsop.maxpool,
        "minpool": tsop.minpool,
    }

    @tvm.register_func("toc_callback_bang_postproc", override=True)
    def toc_callback_bang_postproc(code):
        tvm._ffi.registry.remove_global_func("toc_callback_bang_postproc")
        if not os.path.exists(file):
            with open(file, "w", encoding="utf-8") as f:
                f.write(code)
        code = open(file, encoding="utf-8").read()

        code = code.replace(
            "void " + op_name + "(", "void " + op_name + "_kernel0("
        )

        return code

    def run_cpu(data, kernel_stride, op):
        return _op2np[op](data, kernel_stride)

    input0 = tsop.tensor(shape, dtype=bangpy.float32, name="input0")
    # Describ Computation
    result = _opTensorOp[op_name](input0, kh, kw, sh, sw)
    # Build ang get executable module
    fmlu = tsop.BuildBANG([input0], [result], "mlu590-h8", kernel_name=op_name)
    # Generate random test data and run on mlu and cpu

    def generate_data(shape, dtype):
        return np.random.uniform(size=shape).astype(dtype)

    data0 = generate_data(shape, "float32")
    cpu_output = run_cpu(data0, kernel_stride, op_name)
    result_np = np.zeros(shape=cpu_output.shape, dtype="float32")

    dev = bangpy.device(0)
    data_dev = bangpy.Array(data0, dev)
    result_arr = bangpy.Array(result_np, dev)

    fmlu(data_dev, result_arr)
    # Compare
    bangpy.assert_allclose(result_arr.numpy(), cpu_output, 0.1, 0)
    print("验证通过！")

    evaluator = fmlu.time_evaluator(number=100, repeat=1, min_repeat_ms=0)
    cost = evaluator(data_dev, result_arr).mean * 1e3
    print(f"{name} execution time: {cost} ms")


def perf_bmm(name, file, shape_A, shape_B, shape_C):
    # 创建随机张量
    A = te.placeholder(shape_A, name="A", dtype="float32")
    B = te.placeholder(shape_B, name="B", dtype="float32")
    op_name = name.split("_")[0]

    @tvm.register_func
    def toc_callback_bang_postproc(code):
        tvm._ffi.registry.remove_global_func("toc_callback_bang_postproc")
        if not os.path.exists(file):
            with open(file, "w", encoding="utf-8") as f:
                f.write(code)
        code = open(file, encoding="utf-8").read()

        code = code.replace(
            "void " + op_name + "(", "void " + op_name + "_kernel0("
        )

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
    a = tvm.nd.array(np.random.rand(*shape_A).astype("float32"), dev)
    b = tvm.nd.array(np.random.rand(*shape_B).astype("float32"), dev)
    c = tvm.nd.array(np.random.rand(*shape_C).astype("float32"), dev)
    with toc.build_config(env):
        f = toc.build(s, [A, B, C], name=op_name)
    f(a, b, c)
    time_f = f.time_evaluator(op_name, dev, number=20, repeat=100)
    cost = time_f(a, b, c)
    print(f"{name} execution time: {cost.mean * 1000} ms")
    func_name = "toc_callback_bang_postproc"
    tvm._ffi.registry.remove_global_func(func_name)


def perf_activation(name, file, shape):
    op_name = name.split("_")[0]
    A = te.placeholder(shape, dtype="float32", name="A")

    @tvm.register_func
    def toc_callback_bang_postproc(code):
        tvm._ffi.registry.remove_global_func("toc_callback_bang_postproc")
        if not os.path.exists(file):
            with open(file, "w", encoding="utf-8") as f:
                f.write(code)
        code = open(file, encoding="utf-8").read()

        code = code.replace(
            "void " + op_name + "(", "void " + op_name + "_kernel0("
        )

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
    a = tvm.nd.array(np.random.rand(*shape).astype("float32"), dev)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), dev)
    with toc.build_config(env):
        f = toc.build(s, [A, B], name=op_name)
    f(a, b)
    time_f = f.time_evaluator(op_name, dev, number=20, repeat=100)
    cost = time_f(a, b)
    print(f"{name} execution time: {cost.mean * 1000} ms")
    func_name = "toc_callback_bang_postproc"
    tvm._ffi.registry.remove_global_func(func_name)


def perf_conv2d(name, file, shape, kernel, output_shape, stride, pad):

    op_name = "conv2d"

    @tvm.register_func
    def toc_callback_bang_postproc(code):
        tvm._ffi.registry.remove_global_func("toc_callback_bang_postproc")
        if not os.path.exists(file):
            with open(file, "w", encoding="utf-8") as f:
                f.write(code)
        code = open(file, encoding="utf-8").read()

        code = code.replace(
            "void " + op_name + "(", "void " + op_name + "_kernel0("
        )
        print(code)
        return code

    # generate data
    data_np = np.random.uniform(low=1.0, high=2.0, size=shape).astype(
        "float32"
    )
    kernel_np = np.random.uniform(low=1.0, high=2.0, size=kernel).astype(
        "float32"
    )
    # cpu compute
    A = te.placeholder(data_shape, dtype="float32", name="A")
    B = te.placeholder(kernel_shape, dtype="float32", name="B")

    def conv(A, B, C):
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
        lambda ins, outs: conv(ins[0], ins[1], outs[0]),
        name="conv2d",
        dtype="float32",
    )

    s = te.create_schedule(C.op)

    dev = tvm.device("bang", 0)
    data_dev = tvm.nd.array(data_np, dev)
    kernel_dev = tvm.nd.array(kernel_np, dev)
    result_np = np.zeros(output_shape, dtype="float32")
    result_dev = tvm.nd.array(result_np, dev)
    with toc.build_config(env):
        func = toc.build(s, [A, B, C], name="conv2d")

    func(data_dev, kernel_dev, result_dev)
    time_f = func.time_evaluator("conv2d", dev, number=20)
    cost = time_f(data_dev, kernel_dev, result_dev).mean * 1e3
    print(f"{name} execution time: {cost} ms")
    func_name = "toc_callback_bang_postproc"
    tvm._ffi.registry.remove_global_func(func_name)


def perf_conv2d_nchw(name, file, shape, kernel, output_shape, stride, pad):
    @tvm.register_func
    def toc_callback_bang_postproc(code, target):
        code = open(file).read()

        code = code.replace("conv2d" + "(", "conv2d" + "_kernel(")

        return code

    # generate data
    data_np = np.random.uniform(low=1.0, high=2.0, size=shape).astype(
        "float32"
    )
    kernel_np = np.random.uniform(low=1.0, high=2.0, size=kernel).astype(
        "float32"
    )
    # cpu compute
    A = te.placeholder(data_shape, dtype="float32", name="A")
    B = te.placeholder(kernel_shape, dtype="float32", name="B")

    def conv(A, B, C):
        n = A.shape[0]
        prod = np.prod(A.shape[:-1])
        ib = tvm.tir.ir_builder.create()
        tx = te.thread_axis("threadIdx.x")
        bx = te.thread_axis("blockIdx.x")
        if prod < 4:
            max_threads = prod
            ib.scope_attr(tx, "thread_extent", max_threads)

        else:
            max_threads = 4
            max_block = (
                prod // max_threads
                if prod % max_threads == 0
                else prod // max_threads + 1
            )
            ib.scope_attr(tx, "thread_extent", max_threads)
            ib.scope_attr(bx, "thread_extent", max_block)

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
        lambda ins, outs: conv(ins[0], ins[1], outs[0]),
        name="conv2d",
        dtype="float32",
    )

    s = te.create_schedule(C.op)

    dev = tvm.device("bang", 0)
    data_dev = tvm.nd.array(data_np, dev)
    kernel_dev = tvm.nd.array(kernel_np, dev)
    result_np = np.zeros(output_shape, dtype="float32")
    result_dev = tvm.nd.array(result_np, dev)
    with toc.build_config(env):
        func = toc.build(s, [A, B, C], name="conv2d")
    func(data_dev, kernel_dev, result_dev)
    time_f = func.time_evaluator("conv2d", dev, number=20)
    cost = time_f(data_dev, kernel_dev, result_dev).mean * 1e3
    print(f"{name} execution time: {cost} ms")
    func_name = "toc_callback_bang_postproc"
    tvm._ffi.registry.remove_global_func(func_name)


def perf_gemv(name, file, shape, kernel_shape, output_shape):
    op_name = name.split("_")[0]
    A = te.placeholder(shape, dtype="float32", name="A")
    B = te.placeholder(kernel_shape, dtype="float32", name="B")

    tvm.tir.decl_buffer(A.shape, "float32", "A_buf")
    tvm.tir.decl_buffer(B.shape, "float32", "B_buf")
    tvm.tir.decl_buffer(output_shape, "float32", "C_buf")

    @tvm.register_func
    def toc_callback_bang_postproc(code):
        tvm._ffi.registry.remove_global_func("toc_callback_bang_postproc")
        if not os.path.exists(file):
            with open(file, "w", encoding="utf-8") as f:
                f.write(code)
        code = open(file, encoding="utf-8").read()

        code = code.replace(
            "void " + op_name + "(", "void " + op_name + "_kernel0("
        )

        return code

    def test_gemv(A, B, C):
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
        lambda ins, outs: test_gemv(ins[0], ins[1], outs[0]),
        name=op_name,
        dtype="float32",
    )

    s = te.create_schedule(C.op)

    dev = tvm.device("bang", 0)
    a = tvm.nd.array(np.random.rand(*shape).astype("float32"), dev)
    b = tvm.nd.array(np.random.rand(*kernel_shape).astype("float32"), dev)
    c = tvm.nd.array(np.random.rand(*output_shape).astype("float32"), dev)
    with toc.build_config(env):
        f = toc.build(s, [A, B, C], name=op_name)
    f(a, b, c)
    time_f = f.time_evaluator(op_name, dev, number=20, repeat=100)
    cost = time_f(a, b, c)
    print(f"{name} execution time: {cost.mean * 1000} ms")
    func_name = "toc_callback_bang_postproc"
    tvm._ffi.registry.remove_global_func(func_name)


def perf_conv1d(name, file, shape, kernel_shape, output_shape):
    op_name = name.split("_")[0]
    A = te.placeholder(shape, dtype="float32", name="A")
    B = te.placeholder(kernel_shape, dtype="float32", name="B")

    tvm.tir.decl_buffer(A.shape, "float32", "A_buf")
    tvm.tir.decl_buffer(B.shape, "float32", "B_buf")
    tvm.tir.decl_buffer(output_shape, "float32", "C_buf")

    @tvm.register_func
    def toc_callback_bang_postproc(code):
        tvm._ffi.registry.remove_global_func("toc_callback_bang_postproc")
        if not os.path.exists(file):
            with open(file, "w", encoding="utf-8") as f:
                f.write(code)
        code = open(file, encoding="utf-8").read()

        code = code.replace(
            "void " + op_name + "(", "void " + op_name + "_kernel0("
        )

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
    a = tvm.nd.array(np.random.rand(*shape).astype("float32"), dev)
    b = tvm.nd.array(np.random.rand(*kernel_shape).astype("float32"), dev)
    c = tvm.nd.array(np.random.rand(*output_shape).astype("float32"), dev)
    with toc.build_config(env):
        f = toc.build(s, [A, B, C], name=op_name)
    f(a, b, c)
    time_f = f.time_evaluator(op_name, dev, number=20, repeat=100)
    cost = time_f(a, b, c)
    print(f"{name} execution time: {cost.mean * 1000} ms")
    func_name = "toc_callback_bang_postproc"
    tvm._ffi.registry.remove_global_func(func_name)


def perf_depthwise_conv2d(name, file, shape, kernel_shape, output_shape):
    op_name = "depthwise_convolution"
    A = te.placeholder(shape, dtype="float32", name="A")
    B = te.placeholder(kernel_shape, dtype="float32", name="B")

    tvm.tir.decl_buffer(A.shape, "float32", "A_buf")
    tvm.tir.decl_buffer(B.shape, "float32", "B_buf")
    tvm.tir.decl_buffer(output_shape, "float32", "C_buf")

    @tvm.register_func
    def toc_callback_bang_postproc(code):
        tvm._ffi.registry.remove_global_func("toc_callback_bang_postproc")
        if not os.path.exists(file):
            with open(file, "w", encoding="utf-8") as f:
                f.write(code)
        code = open(file, encoding="utf-8").read()

        code = code.replace(
            "void " + op_name + "(", "void " + op_name + "_kernel0("
        )

        return code

    def test_depthwise_conv2d(A, B, C):
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
        lambda ins, outs: test_depthwise_conv2d(ins[0], ins[1], outs[0]),
        name=op_name,
        dtype="float32",
    )

    s = te.create_schedule(C.op)

    dev = tvm.device("bang", 0)
    a = tvm.nd.array(np.random.rand(*shape).astype("float32"), dev)
    b = tvm.nd.array(np.random.rand(*kernel_shape).astype("float32"), dev)
    c = tvm.nd.array(np.random.rand(*output_shape).astype("float32"), dev)
    with toc.build_config(env):
        f = toc.build(s, [A, B, C], name=op_name)
    f(a, b, c)
    time_f = f.time_evaluator(op_name, dev, number=20, repeat=100)
    cost = time_f(a, b, c)
    print(f"{name} execution time: {cost.mean * 1000} ms")
    func_name = "toc_callback_bang_postproc"
    tvm._ffi.registry.remove_global_func(func_name)


def perf_layernorm(name, file, shape):
    op_name = name.split("_")[0]
    A = te.placeholder(shape, dtype="float32", name="A")
    B = te.placeholder(shape[-1:], dtype="float32", name="B")
    C = te.placeholder(shape[-1:], dtype="float32", name="C")

    tvm.tir.decl_buffer(A.shape, "float32", "A_buf")
    tvm.tir.decl_buffer(B.shape, "float32", "B_buf")
    tvm.tir.decl_buffer(C.shape, "float32", "C_buf")
    tvm.tir.decl_buffer(shape, "float32", "D_buf")

    @tvm.register_func
    def toc_callback_bang_postproc(code):
        tvm._ffi.registry.remove_global_func("toc_callback_bang_postproc")
        if not os.path.exists(file):
            with open(file, "w", encoding="utf-8") as f:
                f.write(code)
        code = open(file, encoding="utf-8").read()

        code = code.replace(
            "void " + op_name + "(", "void " + op_name + "_kernel0("
        )

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
    a = tvm.nd.array(np.random.rand(*shape).astype("float32"), dev)
    b = tvm.nd.array(np.random.rand(*shape[-1:]).astype("float32"), dev)
    c = tvm.nd.array(np.random.rand(*shape[-1:]).astype("float32"), dev)
    d = tvm.nd.array(np.random.rand(*shape).astype("float32"), dev)
    with toc.build_config(env):
        f = toc.build(s, [A, B, C, D], name=op_name)
    f(a, b, c, d)
    time_f = f.time_evaluator(op_name, dev, number=20, repeat=100)
    cost = time_f(a, b, c, d)
    print(f"{name} execution time: {cost.mean * 1000} ms")
    func_name = "toc_callback_bang_postproc"
    tvm._ffi.registry.remove_global_func(func_name)


def perf_rmsnorm(name, file, shape):
    op_name = name.split("_")[0]
    A = te.placeholder(shape, dtype="float32", name="A")

    @tvm.register_func
    def toc_callback_bang_postproc(code):
        tvm._ffi.registry.remove_global_func("toc_callback_bang_postproc")
        if not os.path.exists(file):
            with open(file, "w", encoding="utf-8") as f:
                f.write(code)
        code = open(file, encoding="utf-8").read()

        code = code.replace(
            "void " + op_name + "(", "void " + op_name + "_kernel0("
        )

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
    a = tvm.nd.array(np.random.rand(*shape).astype("float32"), dev)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), dev)
    with toc.build_config(env):
        f = toc.build(s, [A, B], name=op_name)
    f(a, b)
    time_f = f.time_evaluator(op_name, dev, number=20, repeat=100)
    cost = time_f(a, b)
    print(f"{name} execution time: {cost.mean * 1000} ms")
    func_name = "toc_callback_bang_postproc"
    tvm._ffi.registry.remove_global_func(func_name)


def perf_deformable(name, file, shape):
    op_name = "deformable"
    N, M, D = shape[:3]
    Lq, L, P = shape[3:]
    shapes = torch.as_tensor(
        [[84, 117], [42, 59], [21, 30], [11, 15]], dtype=torch.long
    )
    level_start_index = torch.cat(
        (shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1])
    )
    S = sum([(H * W).item() for H, W in shapes])

    A = te.placeholder([N, S, M, D], dtype="float32", name="A")
    B = te.placeholder([N, Lq, M, L, P, 2], dtype="float32", name="B")
    C = te.placeholder([N, Lq, M, L, P], dtype="float32", name="C")
    shape_pl = te.placeholder([4, 2], dtype="int32", name="shape")
    output_shape = [N, Lq, M * D]

    tvm.tir.decl_buffer(A.shape, "float32", "A_buf")
    tvm.tir.decl_buffer(shape_pl.shape, "int32", "A_buf")
    tvm.tir.decl_buffer(B.shape, "float32", "B_buf")
    tvm.tir.decl_buffer(C.shape, "float32", "C_buf")
    tvm.tir.decl_buffer(output_shape, "float32", "D_buf")

    @tvm.register_func
    def toc_callback_bang_postproc(code):
        tvm._ffi.registry.remove_global_func("toc_callback_bang_postproc")
        if not os.path.exists(file):
            with open(file, "w", encoding="utf-8") as f:
                f.write(code)
        code = open(file, encoding="utf-8").read()

        code = code.replace(
            "void " + op_name + "(", "void " + op_name + "_kernel0("
        )

        return code

    def test_deformable(A, B, C, D, E):
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

    out_D = te.extern(
        output_shape,
        [A, shape_pl, B, C],
        lambda ins, outs: test_deformable(
            ins[0], ins[1], ins[2], ins[3], outs[0]
        ),
        name=op_name,
        dtype="float32",
    )

    s = te.create_schedule(out_D.op)

    dev = tvm.device("bang", 0)
    a = tvm.nd.array(np.random.rand(N, S, M, D).astype("float32"), dev)
    b = tvm.nd.array(np.random.rand(4, 2).astype("int32"), dev)
    c = tvm.nd.array(np.random.rand(N, Lq, M, L, P, 2).astype("float32"), dev)
    d = tvm.nd.array(np.random.rand(N, Lq, M, L, P).astype("float32"), dev)
    e = tvm.nd.array(np.random.rand(N, Lq, M * D).astype("float32"), dev)
    with toc.build_config(env):
        f = toc.build(s, [A, shape_pl, B, C, out_D], name=op_name)
    f(a, b, c, d, e)
    time_f = f.time_evaluator(op_name, dev, number=20, repeat=100)
    cost = time_f(a, b, c, d, e)
    print(f"{name} execution time: {cost.mean * 1000} ms")
    func_name = "toc_callback_bang_postproc"
    tvm._ffi.registry.remove_global_func(func_name)


def perf_scaled_dot_product_attention(name, file, shape):
    op_name = name.split("_")[0]
    A = te.placeholder(shape, dtype="float32", name="A")
    B = te.placeholder(shape, dtype="float32", name="B")
    C = te.placeholder(shape, dtype="float32", name="C")

    A_buff = tvm.tir.decl_buffer(A.shape, "float32", "A_buf")
    B_buff = tvm.tir.decl_buffer(B.shape, "float32", "B_buf")
    C_buff = tvm.tir.decl_buffer(C.shape, "float32", "C_buf")
    D_buff = tvm.tir.decl_buffer(shape, "float32", "D_buf")

    @tvm.register_func
    def toc_callback_bang_postproc(code):
        tvm._ffi.registry.remove_global_func("toc_callback_bang_postproc")
        if not os.path.exists(file):
            with open(file, "w", encoding="utf-8") as f:
                f.write(code)
        code = open(file, encoding="utf-8").read()

        code = code.replace(
            "void " + op_name + "(", "void " + op_name + "_kernel0("
        )

        return code

    def test_scaled_dot_product_attention(
        A, B, C, D, seq_len, num_heads, head_dim
    ):
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
    func_name = "toc_callback_bang_postproc"
    tvm._ffi.registry.remove_global_func(func_name)


if __name__ == "__main__":
    files = glob.glob(
        os.path.join(os.getcwd(), "benchmark/data/mlu_code_test/add_*.mlu")
    )
    counter = 0

    for file in files:
        base_name = os.path.basename(file)
        name = base_name.split("_")[0]
        if name == "add" or name == "sign":
            shapes = base_name.split(".")[0]
            shape = [int(intg) for intg in shapes.split("_")[1:]]
            import time

            t1 = time.time()
            perf_elementwise(base_name, file, shape)
            t2 = time.time()
            print(f"(eval time {t2-t1}")
        elif name in ["avgpool", "maxpool", "minpool", "sumpool"]:
            shape = base_name.split("_")[1:5]
            shape = [int(intg) for intg in shape]
            kernel_stride = base_name.split(".")[0].split("_")[5:]
            kernel_stride = [int(intg) for intg in kernel_stride]
            perf_pooling(
                base_name, file, shape, kernel_stride[:2], kernel_stride[2:]
            )

        elif name == "bmm":
            shapes = base_name.split(".")[0]
            shape = [int(intg) for intg in shapes.split("_")[1:]]
            batch_size, matrix_dim_i, matrix_dim_j, matrix_dim_k = shape
            shape_A = [batch_size, matrix_dim_i, matrix_dim_j]
            shape_B = [batch_size, matrix_dim_k, matrix_dim_j]
            shape_C = [batch_size, matrix_dim_i, matrix_dim_k]
            perf_bmm(name, file, shape_A, shape_B, shape_C)

        elif name == "gemm":
            shapes = base_name.split(".")[0]
            shape = [int(intg) for intg in shapes.split("_")[1:]]
            shape_A = [1, shape[0], shape[1]]
            shape_B = [1, shape[2], shape[1]]
            shape_C = [1, shape[0], shape[2]]
            perf_bmm(name, file, shape_A, shape_B, shape_C)

        elif name in ["relu", "sigmoid", "gelu", "softmax"]:
            shapes = base_name.split(".")[0]
            shape = [int(intg) for intg in shapes.split("_")[1:]]
            perf_activation(base_name, file, shape)

        elif name == "conv2d":
            data_shape = base_name.split("_")[1:5]
            data_shape = [int(intg) for intg in data_shape]

            kernel_shape = base_name.split("_")[5:9]
            kernel_shape = [int(intg) for intg in kernel_shape]
            stride_h = stride_w = int(base_name.split("_")[9])
            pad_h = pad_w = int(base_name.split("_")[10].replace(".mlu", ""))

            batch_size, input_height, input_width, input_channel = data_shape
            output_channel, kernel_height, kernel_width, _ = kernel_shape
            out_height = int(
                (input_height + np.sum(pad_h) - kernel_height) / stride_h + 1
            )
            out_width = int(
                (input_width + np.sum(pad_w) - kernel_width) / stride_w + 1
            )
            output_shape = [batch_size, out_height, out_width, output_channel]

            perf_conv2d(
                name,
                file,
                data_shape,
                kernel_shape,
                output_shape,
                stride_h,
                pad_h,
            )

        elif name == "conv2dnchw_1":
            data_shape = base_name.split("_")[1:5]
            data_shape = [int(intg) for intg in data_shape]
            kernel_shape = base_name.split("_")[5:9]
            kernel_shape = [int(intg) for intg in kernel_shape]
            stride_h = stride_w = int(base_name.split(".")[0].split("_")[9])
            pad = int(base_name.split(".")[0].split("_")[10])

            perf_conv2d_nchw(
                base_name,
                data_shape,
                kernel_shape[1],
                kernel_shape[0],
                kernel_shape[2],
                stride_h,
                pad,
            )

        elif name == "gemv":
            shapes = base_name.split(".")[0]
            shape = [int(intg) for intg in shapes.split("_")[1:]]
            kernel_shape = [shape[1]]
            output_shape = [shape[0]]
            perf_gemv(base_name, file, shape, kernel_shape, output_shape)

        elif name == "conv1d":
            shapes = base_name.split(".")[0]
            shape = [int(intg) for intg in shapes.split("_")[1:]]
            shape = [shape[1]]
            kernel_shape = [3]
            output_shape = [shape[0]]
            perf_conv1d(base_name, file, shape, kernel_shape, output_shape)

        elif name == "depthwiseconv":
            shapes = base_name.split(".")[0]
            shape = [int(intg) for intg in shapes.split("_")[1:]]
            input_height, kernel_size, input_channels = (
                shape[0],
                shape[1],
                shape[2],
            )
            shape = [input_height, input_height, input_channels]
            kernel_shape = [kernel_size, kernel_size, input_channels]
            # Calculate the output tensor shape
            output_height = input_height - kernel_size + 1
            output_width = input_height - kernel_size + 1
            output_shape = [output_height, output_width, input_channels]
            perf_depthwise_conv2d(
                base_name, file, shape, kernel_shape, output_shape
            )

        elif name == "layernorm":
            shapes = base_name.split(".")[0]
            shape = [int(intg) for intg in shapes.split("_")[1:]]
            perf_layernorm(base_name, file, shape)

        elif name == "rmsnorm":
            shapes = base_name.split(".")[0]
            shape = [int(intg) for intg in shapes.split("_")[1:]]
            perf_rmsnorm(base_name, file, shape)

        elif name == "deformable":
            shapes = base_name.split(".")[0]
            shape = [int(intg) for intg in shapes.split("_")[1:]]
            perf_deformable(base_name, file, shape)

        elif name == "mha":
            shapes = base_name.split(".")[0]
            shape = [int(intg) for intg in shapes.split("_")[1:]]
            perf_scaled_dot_product_attention(base_name, file, shape)
