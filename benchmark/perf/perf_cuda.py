import glob
import os

import numpy as np
import torch
import tvm
import tvm.topi.testing
from tvm import te, topi
from tvm.topi.utils import get_const_tuple


def perf_elementwise(name, file, shape):
    x = te.placeholder(shape, name="x", dtype="float32")
    y = te.placeholder(shape, name="y", dtype="float32")
    op_name = name.split("_")[0]

    @tvm.register_func
    def tvm_callback_cuda_postproc(code, target):
        code = open(file).read()
        code = code.split("extern")[0]
        code = code.replace(op_name + "(", op_name + "_kernel(")
        code = 'extern "C" ' + code
        return code

    dev = tvm.cuda(0)
    if op_name == "add":
        C = topi.add(x, y)
        with tvm.target.Target("cuda"):
            s = tvm.topi.testing.get_elemwise_schedule("cuda")(C)
        foo = tvm.build(s, [x, y, C], "cuda", name=op_name)

        a = tvm.nd.array(np.random.rand(*shape).astype("float32"), dev)
        b = tvm.nd.array(np.random.rand(*shape).astype("float32"), dev)
        c = tvm.nd.array(np.random.rand(*shape).astype("float32"), dev)
        foo(a, b, c)
        time_f = foo.time_evaluator(op_name, dev, number=20, repeat=100)
        cost = time_f(a, b, c)
        print(f"{name} execution time: {cost.mean * 1000} ms")
        func_name = "tvm_callback_cuda_postproc"
        tvm._ffi.registry.remove_global_func(func_name)

    elif op_name == "sign":
        C = topi.sign(x)
        with tvm.target.Target("cuda"):
            s = tvm.topi.testing.get_injective_schedule("cuda")(C)

        foo = tvm.build(s, [x, C], "cuda", name=op_name)
        time_f = foo.time_evaluator(op_name, dev, number=20, repeat=100)
        a = tvm.nd.array(np.random.rand(*shape).astype("float32"), dev)
        c = tvm.nd.array(np.random.rand(*shape).astype("float32"), dev)
        foo(a, c)
        cost = time_f(a, c)
        print(f"{name} execution time: {cost.mean * 1000} ms")
        func_name = "tvm_callback_cuda_postproc"
        tvm._ffi.registry.remove_global_func(func_name)


def perf_pooling(name, file, shape, kernel, stride):
    op_name = name.split("_")[0]
    poolType = {
        "avgpool": "avg",
        "sumpool": "avg",
        "maxpool": "max",
        "minpool": "max",
    }

    @tvm.register_func
    def tvm_callback_cuda_postproc(code, target):
        code = open(file).read()
        code = code.split("extern")[0]
        code = code.replace(op_name + "(", op_name + "_kernel(")
        code = 'extern "C" ' + code
        return code

    A = te.placeholder(shape, name="A", dtype="float32")
    B = topi.nn.pool2d(
        A,
        kernel=kernel,
        stride=stride,
        dilation=[1, 1],
        padding=[0, 0, 0, 0],
        pool_type=poolType[op_name],
        ceil_mode=False,
        layout="NHWC",
        count_include_pad=False,
    )
    with tvm.target.Target("cuda"):
        s = topi.cuda.schedule_pool(B, layout="NHWC")

    dev = tvm.cuda(0)

    a = tvm.nd.array(np.random.rand(*shape).astype("float32"), dev)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype="float32"), dev)
    f = tvm.build(s, [A, B], "cuda", name=op_name)
    f(a, b)
    time_f = f.time_evaluator(op_name, dev, number=20, repeat=100)
    cost = time_f(a, b)
    print(f"{name} execution time: {cost.mean * 1000} ms")
    func_name = "tvm_callback_cuda_postproc"
    tvm._ffi.registry.remove_global_func(func_name)


def perf_bmm(name, file, shape_A, shape_B, shape_C):
    # 创建随机张量
    A = te.placeholder(shape_A, name="A", dtype="float32")
    B = te.placeholder(shape_B, name="B", dtype="float32")
    op_name = name.split("_")[0]

    @tvm.register_func
    def tvm_callback_cuda_postproc(code, target):
        code = open(file).read()
        code = code.split("extern")[0]
        code = code.replace(op_name + "(", op_name + "_kernel(")
        code = 'extern "C" ' + code
        return code

    C = topi.cuda.batch_matmul(A, B)
    with tvm.target.Target("cuda"):
        s = topi.cuda.schedule_batch_matmul(C)

    dev = tvm.cuda(0)

    a = tvm.nd.array(np.random.rand(*shape_A).astype("float32"), dev)
    b = tvm.nd.array(np.random.rand(*shape_B).astype("float32"), dev)
    c = tvm.nd.array(np.random.rand(*shape_C).astype("float32"), dev)
    f = tvm.build(s, [A, B, C], "cuda", name=op_name)
    f(a, b, c)
    time_f = f.time_evaluator(op_name, dev, number=20, repeat=100)
    cost = time_f(a, b, c)
    print(f"{name} execution time: {cost.mean * 1000} ms")
    func_name = "tvm_callback_cuda_postproc"
    tvm._ffi.registry.remove_global_func(func_name)


def perf_activation(name, file, shape):
    op_name = name.split("_")[0]
    A = te.placeholder(shape, dtype="float32", name="A")

    @tvm.register_func
    def tvm_callback_cuda_postproc(code, target):
        code = open(file).read()
        code = code.split("extern")[0]
        code = code.replace(op_name + "(", op_name + "_kernel(")
        code = 'extern "C" ' + code
        return code

    def test_activation(A, B):
        n = A.shape[0]
        prod = np.prod(A.shape[:-1])
        ib = tvm.tir.ir_builder.create()
        tx = te.thread_axis("threadIdx.x")
        bx = te.thread_axis("blockIdx.x")
        if prod < 1024:
            max_threads = prod
            ib.scope_attr(tx, "thread_extent", max_threads)

        else:
            max_threads = 1024
            max_block = (
                prod // max_threads
                if prod % max_threads == 0
                else prod // max_threads + 1
            )
            ib.scope_attr(tx, "thread_extent", max_threads)
            ib.scope_attr(bx, "thread_extent", max_block)

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
    with tvm.target.Target("cuda"):
        s = te.create_schedule(B.op)

    dev = tvm.cuda(0)
    a = tvm.nd.array(np.random.rand(*shape).astype("float32"), dev)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), dev)
    f = tvm.build(s, [A, B], "cuda", name=op_name)
    f(a, b)
    time_f = f.time_evaluator(op_name, dev, number=20, repeat=100)
    cost = time_f(a, b)
    print(f"{name} execution time: {cost.mean * 1000} ms")
    func_name = "tvm_callback_cuda_postproc"
    tvm._ffi.registry.remove_global_func(func_name)


def perf_conv2d(name, file, shape, kernel, output_shape, stride, pad):
    @tvm.register_func
    def tvm_callback_cuda_postproc(code, target):
        code = open(file).read()
        code = code.split("extern")[0]
        code = code.replace("conv2d" + "(", "conv2d" + "_kernel(")
        code = 'extern "C" ' + code
        return code

    # generate data
    data_np = np.random.uniform(low=1.0, high=2.0, size=shape).astype("float32")
    kernel_np = np.random.uniform(low=1.0, high=2.0, size=kernel).astype("float32")
    # cpu compute
    A = te.placeholder(data_shape, dtype="float32", name="A")
    B = te.placeholder(kernel_shape, dtype="float32", name="B")

    def conv(A, B, C):
        n = A.shape[0]
        prod = np.prod(A.shape[:-1])
        ib = tvm.tir.ir_builder.create()
        tx = te.thread_axis("threadIdx.x")
        bx = te.thread_axis("blockIdx.x")
        if prod < 1024:
            max_threads = prod
            ib.scope_attr(tx, "thread_extent", max_threads)

        else:
            max_threads = 1024
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

    with tvm.target.Target("cuda"):
        s = te.create_schedule(C.op)

    dev = tvm.cuda(0)
    data_dev = tvm.nd.array(data_np, dev)
    kernel_dev = tvm.nd.array(kernel_np, dev)
    result_np = np.zeros(output_shape, dtype="float32")
    result_dev = tvm.nd.array(result_np, dev)
    func = tvm.build(s, [A, B, C], "cuda", name="conv2d")
    func(data_dev, kernel_dev, result_dev)
    time_f = func.time_evaluator("conv2d", dev, number=20)
    cost = time_f(data_dev, kernel_dev, result_dev).mean * 1e3
    print(f"{name} execution time: {cost} ms")
    func_name = "tvm_callback_cuda_postproc"
    tvm._ffi.registry.remove_global_func(func_name)


def perf_conv2d_nchw(name, file, shape, kernel, output_shape, stride, pad):
    @tvm.register_func
    def tvm_callback_cuda_postproc(code, target):
        code = open(file).read()
        code = code.split("extern")[0]
        code = code.replace("conv2d" + "(", "conv2d" + "_kernel(")
        code = 'extern "C" ' + code
        return code

    # generate data
    data_np = np.random.uniform(low=1.0, high=2.0, size=shape).astype("float32")
    kernel_np = np.random.uniform(low=1.0, high=2.0, size=kernel).astype("float32")
    # cpu compute
    A = te.placeholder(data_shape, dtype="float32", name="A")
    B = te.placeholder(kernel_shape, dtype="float32", name="B")

    def conv(A, B, C):
        n = A.shape[0]
        prod = np.prod(A.shape[:-1])
        ib = tvm.tir.ir_builder.create()
        tx = te.thread_axis("threadIdx.x")
        bx = te.thread_axis("blockIdx.x")
        if prod < 1024:
            max_threads = prod
            ib.scope_attr(tx, "thread_extent", max_threads)

        else:
            max_threads = 1024
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

    with tvm.target.Target("cuda"):
        s = te.create_schedule(C.op)

    dev = tvm.cuda(0)
    data_dev = tvm.nd.array(data_np, dev)
    kernel_dev = tvm.nd.array(kernel_np, dev)
    result_np = np.zeros(output_shape, dtype="float32")
    result_dev = tvm.nd.array(result_np, dev)
    func = tvm.build(s, [A, B, C], "cuda", name="conv2d")
    func(data_dev, kernel_dev, result_dev)
    time_f = func.time_evaluator("conv2d", dev, number=20)
    cost = time_f(data_dev, kernel_dev, result_dev).mean * 1e3
    print(f"{name} execution time: {cost} ms")
    func_name = "tvm_callback_cuda_postproc"
    tvm._ffi.registry.remove_global_func(func_name)


def perf_gemv(name, file, shape, kernel_shape, output_shape):
    op_name = name.split("_")[0]
    A = te.placeholder(shape, dtype="float32", name="A")
    B = te.placeholder(kernel_shape, dtype="float32", name="B")

    A_buff = tvm.tir.decl_buffer(A.shape, "float32", "A_buf")
    B_buff = tvm.tir.decl_buffer(B.shape, "float32", "B_buf")
    C_buff = tvm.tir.decl_buffer(output_shape, "float32", "C_buf")

    @tvm.register_func
    def tvm_callback_cuda_postproc(code, target):
        code = open(file).read()
        code = code.split("extern")[0]
        code = code.replace(op_name + "(", op_name + "_kernel(")
        code = 'extern "C" ' + code
        return code

    def test_gemv(A, B, C):
        n = A.shape[0]
        prod = np.prod(A.shape[:-1])
        ib = tvm.tir.ir_builder.create()
        tx = te.thread_axis("threadIdx.x")
        bx = te.thread_axis("blockIdx.x")
        ib.scope_attr(tx, "thread_extent", 1024)
        ib.scope_attr(bx, "thread_extent", 256)

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
    with tvm.target.Target("cuda"):
        s = te.create_schedule(C.op)

    dev = tvm.cuda(0)
    a = tvm.nd.array(np.random.rand(*shape).astype("float32"), dev)
    b = tvm.nd.array(np.random.rand(*kernel_shape).astype("float32"), dev)
    c = tvm.nd.array(np.random.rand(*output_shape).astype("float32"), dev)
    f = tvm.build(s, [A, B, C], "cuda", name=op_name)
    f(a, b, c)
    time_f = f.time_evaluator(op_name, dev, number=20, repeat=100)
    cost = time_f(a, b, c)
    print(f"{name} execution time: {cost.mean * 1000} ms")
    func_name = "tvm_callback_cuda_postproc"
    tvm._ffi.registry.remove_global_func(func_name)


def perf_conv1d(name, file, shape, kernel_shape, output_shape):
    op_name = name.split("_")[0]
    A = te.placeholder(shape, dtype="float32", name="A")
    B = te.placeholder(kernel_shape, dtype="float32", name="B")

    A_buff = tvm.tir.decl_buffer(A.shape, "float32", "A_buf")
    B_buff = tvm.tir.decl_buffer(B.shape, "float32", "B_buf")
    C_buff = tvm.tir.decl_buffer(output_shape, "float32", "C_buf")

    @tvm.register_func
    def tvm_callback_cuda_postproc(code, target):
        code = open(file).read()
        code = code.split("extern")[0]
        code = code.replace(op_name + "(", op_name + "_kernel(")
        code = 'extern "C" ' + code
        return code

    def test_conv1d(A, B, C):
        n = A.shape[0]
        prod = np.prod(A.shape[:-1])
        ib = tvm.tir.ir_builder.create()
        tx = te.thread_axis("threadIdx.x")
        bx = te.thread_axis("blockIdx.x")
        ib.scope_attr(tx, "thread_extent", 1024)
        ib.scope_attr(bx, "thread_extent", 256)

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
    with tvm.target.Target("cuda"):
        s = te.create_schedule(C.op)

    dev = tvm.cuda(0)
    a = tvm.nd.array(np.random.rand(*shape).astype("float32"), dev)
    b = tvm.nd.array(np.random.rand(*kernel_shape).astype("float32"), dev)
    c = tvm.nd.array(np.random.rand(*output_shape).astype("float32"), dev)
    f = tvm.build(s, [A, B, C], "cuda", name=op_name)
    f(a, b, c)
    time_f = f.time_evaluator(op_name, dev, number=20, repeat=100)
    cost = time_f(a, b, c)
    print(f"{name} execution time: {cost.mean * 1000} ms")
    func_name = "tvm_callback_cuda_postproc"
    tvm._ffi.registry.remove_global_func(func_name)


def perf_depthwise_conv2d(name, file, shape, kernel_shape, output_shape):
    op_name = "depthwise_convolution"
    A = te.placeholder(shape, dtype="float32", name="A")
    B = te.placeholder(kernel_shape, dtype="float32", name="B")

    A_buff = tvm.tir.decl_buffer(A.shape, "float32", "A_buf")
    B_buff = tvm.tir.decl_buffer(B.shape, "float32", "B_buf")
    C_buff = tvm.tir.decl_buffer(output_shape, "float32", "C_buf")

    @tvm.register_func
    def tvm_callback_cuda_postproc(code, target):
        code = open(file).read()
        code = code.split("extern")[0]
        code = code.replace(op_name + "(", op_name + "_kernel(")
        code = 'extern "C" ' + code
        return code

    def test_depthwise_conv2d(A, B, C):
        n = A.shape[0]
        prod = np.prod(A.shape[:-1])
        ib = tvm.tir.ir_builder.create()
        tx = te.thread_axis("threadIdx.x")
        bx = te.thread_axis("blockIdx.x")
        ib.scope_attr(tx, "thread_extent", 1024)
        ib.scope_attr(bx, "thread_extent", 256)

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
    with tvm.target.Target("cuda"):
        s = te.create_schedule(C.op)

    dev = tvm.cuda(0)
    a = tvm.nd.array(np.random.rand(*shape).astype("float32"), dev)
    b = tvm.nd.array(np.random.rand(*kernel_shape).astype("float32"), dev)
    c = tvm.nd.array(np.random.rand(*output_shape).astype("float32"), dev)
    f = tvm.build(s, [A, B, C], "cuda", name=op_name)
    f(a, b, c)
    time_f = f.time_evaluator(op_name, dev, number=20, repeat=100)
    cost = time_f(a, b, c)
    print(f"{name} execution time: {cost.mean * 1000} ms")
    func_name = "tvm_callback_cuda_postproc"
    tvm._ffi.registry.remove_global_func(func_name)


def perf_layernorm(name, file, shape):
    op_name = name.split("_")[0]
    A = te.placeholder(shape, dtype="float32", name="A")
    B = te.placeholder(shape[-1:], dtype="float32", name="B")
    C = te.placeholder(shape[-1:], dtype="float32", name="C")

    A_buff = tvm.tir.decl_buffer(A.shape, "float32", "A_buf")
    B_buff = tvm.tir.decl_buffer(B.shape, "float32", "B_buf")
    C_buff = tvm.tir.decl_buffer(C.shape, "float32", "C_buf")
    D_buff = tvm.tir.decl_buffer(shape, "float32", "D_buf")

    @tvm.register_func
    def tvm_callback_cuda_postproc(code, target):
        code = open(file).read()
        code = code.split("extern")[0]
        code = code.replace(op_name + "(", op_name + "_kernel(")
        code = 'extern "C" ' + code
        return code

    def test_layernorm(A, B, C, D):
        n = A.shape[0]
        prod = np.prod(A.shape[:-1])
        ib = tvm.tir.ir_builder.create()
        tx = te.thread_axis("threadIdx.x")
        bx = te.thread_axis("blockIdx.x")
        ib.scope_attr(tx, "thread_extent", 1024)
        ib.scope_attr(bx, "thread_extent", 256)

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
    with tvm.target.Target("cuda"):
        s = te.create_schedule(D.op)

    dev = tvm.cuda(0)
    a = tvm.nd.array(np.random.rand(*shape).astype("float32"), dev)
    b = tvm.nd.array(np.random.rand(*shape[-1:]).astype("float32"), dev)
    c = tvm.nd.array(np.random.rand(*shape[-1:]).astype("float32"), dev)
    d = tvm.nd.array(np.random.rand(*shape).astype("float32"), dev)
    f = tvm.build(s, [A, B, C, D], "cuda", name=op_name)
    f(a, b, c, d)
    time_f = f.time_evaluator(op_name, dev, number=20, repeat=100)
    cost = time_f(a, b, c, d)
    print(f"{name} execution time: {cost.mean * 1000} ms")
    func_name = "tvm_callback_cuda_postproc"
    tvm._ffi.registry.remove_global_func(func_name)


def perf_rmsnorm(name, file, shape):
    op_name = name.split("_")[0]
    A = te.placeholder(shape, dtype="float32", name="A")

    @tvm.register_func
    def tvm_callback_cuda_postproc(code, target):
        code = open(file).read()
        code = code.split("extern")[0]
        code = code.replace(op_name + "(", op_name + "_kernel(")
        code = 'extern "C" ' + code
        return code

    def test_rmsnorm(A, B):
        n = A.shape[0]
        prod = np.prod(A.shape[:-1])
        ib = tvm.tir.ir_builder.create()
        tx = te.thread_axis("threadIdx.x")
        bx = te.thread_axis("blockIdx.x")
        ib.scope_attr(tx, "thread_extent", 1024)
        ib.scope_attr(bx, "thread_extent", 256)

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
    with tvm.target.Target("cuda"):
        s = te.create_schedule(B.op)

    dev = tvm.cuda(0)
    a = tvm.nd.array(np.random.rand(*shape).astype("float32"), dev)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), dev)
    f = tvm.build(s, [A, B], "cuda", name=op_name)
    f(a, b)
    time_f = f.time_evaluator(op_name, dev, number=20, repeat=100)
    cost = time_f(a, b)
    print(f"{name} execution time: {cost.mean * 1000} ms")
    func_name = "tvm_callback_cuda_postproc"
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

    A_buff = tvm.tir.decl_buffer(A.shape, "float32", "A_buf")
    shape_buffer = tvm.tir.decl_buffer(shape_pl.shape, "int32", "A_buf")
    B_buff = tvm.tir.decl_buffer(B.shape, "float32", "B_buf")
    C_buff = tvm.tir.decl_buffer(C.shape, "float32", "C_buf")
    D_buff = tvm.tir.decl_buffer(output_shape, "float32", "D_buf")

    @tvm.register_func
    def tvm_callback_cuda_postproc(code, target):
        code = open(file).read()
        code = code.split("extern")[0]
        code = code.replace(op_name + "(", op_name + "_kernel(")
        code = 'extern "C" ' + code
        return code

    def test_deformable(A, B, C, D, E):
        n = A.shape[0]
        prod = np.prod(A.shape[:-1])
        ib = tvm.tir.ir_builder.create()
        tx = te.thread_axis("threadIdx.x")
        bx = te.thread_axis("blockIdx.x")
        ib.scope_attr(tx, "thread_extent", 1024)
        ib.scope_attr(bx, "thread_extent", 256)

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
        lambda ins, outs: test_deformable(ins[0], ins[1], ins[2], ins[3], outs[0]),
        name=op_name,
        dtype="float32",
    )
    with tvm.target.Target("cuda"):
        s = te.create_schedule(out_D.op)

    dev = tvm.cuda(0)
    a = tvm.nd.array(np.random.rand(N, S, M, D).astype("float32"), dev)
    b = tvm.nd.array(np.random.rand(4, 2).astype("int32"), dev)
    c = tvm.nd.array(np.random.rand(N, Lq, M, L, P, 2).astype("float32"), dev)
    d = tvm.nd.array(np.random.rand(N, Lq, M, L, P).astype("float32"), dev)
    e = tvm.nd.array(np.random.rand(N, Lq, M * D).astype("float32"), dev)
    f = tvm.build(s, [A, shape_pl, B, C, out_D], "cuda", name=op_name)
    f(a, b, c, d, e)
    time_f = f.time_evaluator(op_name, dev, number=20, repeat=100)
    cost = time_f(a, b, c, d, e)
    print(f"{name} execution time: {cost.mean * 1000} ms")
    func_name = "tvm_callback_cuda_postproc"
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
    def tvm_callback_cuda_postproc(code, target):
        code = open(file).read()
        code = code.split("extern")[0]
        code = code.replace(
            "multi_head_attention" + "(", "multi_head_attention" + "_kernel("
        )
        code = 'extern "C" ' + code
        return code

    def test_scaled_dot_product_attention(A, B, C, D, seq_len, num_heads, head_dim):
        n = A.shape[0]
        prod = np.prod(A.shape[:-1])
        ib = tvm.tir.ir_builder.create()
        tx = te.thread_axis("threadIdx.x")
        bx = te.thread_axis("blockIdx.x")
        ib.scope_attr(tx, "thread_extent", 1024)
        ib.scope_attr(bx, "thread_extent", 256)

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
        name="multi_head_attention",
        in_buffers=[A_buff, B_buff, C_buff],
        out_buffers=[D_buff],
        dtype="float32",
    )
    with tvm.target.Target("cuda"):
        s = te.create_schedule(D.op)

    dev = tvm.cuda(0)
    a = tvm.nd.array(np.random.rand(*shape).astype("float32"), dev)
    b = tvm.nd.array(np.random.rand(*shape).astype("float32"), dev)
    c = tvm.nd.array(np.random.rand(*shape).astype("float32"), dev)
    d = tvm.nd.array(np.random.rand(*shape).astype("float32"), dev)

    f = tvm.build(s, [A, B, C, D], "cuda", name="multi_head_attention")
    f(a, b, c, d)
    time_f = f.time_evaluator("multi_head_attention", dev, number=20, repeat=100)
    cost = time_f(a, b, c, d)
    print(f"{name} execution time: {cost.mean * 1000} ms")
    func_name = "tvm_callback_cuda_postproc"
    tvm._ffi.registry.remove_global_func(func_name)


if __name__ == "__main__":
    files = glob.glob(
        os.path.join(os.getcwd(), "benchmark/data/cuda_code_test/deformable*.cu")
    )
    counter = 0

    for file in files:
        base_name = os.path.basename(file)
        name = base_name.split("_")[0]
        if name == "add" or name == "sign":
            shapes = base_name.split(".")[0]
            shape = [int(intg) for intg in shapes.split("_")[1:]]
            perf_elementwise(base_name, file, shape)

        elif name in ["avgpool", "maxpool", "minpool", "sumpool"]:
            shape = base_name.split("_")[1:5]
            shape = [int(intg) for intg in shape]
            kernel_stride = base_name.split(".")[0].split("_")[5:]
            kernel_stride = [int(intg) for intg in kernel_stride]
            perf_pooling(base_name, file, shape, kernel_stride[:2], kernel_stride[2:])

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
            pad_h = pad_w = int(base_name.split("_")[10].replace(".cu", ""))

            batch_size, input_height, input_width, input_channel = data_shape
            output_channel, kernel_height, kernel_width, _ = kernel_shape
            out_height = int(
                (input_height + np.sum(pad_h) - kernel_height) / stride_h + 1
            )
            out_width = int((input_width + np.sum(pad_w) - kernel_width) / stride_w + 1)
            output_shape = [batch_size, out_height, out_width, output_channel]

            perf_conv2d(
                name, file, data_shape, kernel_shape, output_shape, stride_h, pad_h
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
            input_height, kernel_size, input_channels = shape[0], shape[1], shape[2]
            shape = [input_height, input_height, input_channels]
            kernel_shape = [kernel_size, kernel_size, input_channels]
            # Calculate the output tensor shape
            output_height = input_height - kernel_size + 1
            output_width = input_height - kernel_size + 1
            output_shape = [output_height, output_width, input_channels]
            perf_depthwise_conv2d(base_name, file, shape, kernel_shape, output_shape)

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
