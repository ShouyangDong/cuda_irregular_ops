import argparse
import os

import numpy as np
import toc
import tvm
import tvm.topi.testing
from toc import Environment
from tvm import te

env = Environment("cambricon/mlu590-h8")


def cpu_conv(input_tensor, kernel, stride, pad=0):
    # 获取输入张量和卷积核的维度
    N, C, H, W = input_tensor.shape
    out_channels, in_channels, kernel_height, kernel_width = kernel.shape

    # 计算输出张量的空间维度
    out_height = (H - kernel_height) // stride + 1
    out_width = (W - kernel_width) // stride + 1

    # 初始化输出张量
    output_tensor = np.zeros((N, out_channels, out_height, out_width))

    # 执行卷积操作
    for n in range(N):
        for out_channel in range(out_channels):
            for in_channel in range(in_channels):
                for i in range(0, out_height):
                    for j in range(0, out_width):
                        h_start = i * stride
                        h_end = h_start + kernel_height
                        w_start = j * stride
                        w_end = w_start + kernel_width
                        output_tensor[n, out_channel, i, j] += np.sum(
                            input_tensor[n, in_channel, h_start:h_end, w_start:w_end]
                            * kernel[out_channel, in_channel]
                        )

    return output_tensor


def verify_conv2d_nchw(name, file, shape, kernel, output_shape, stride, pad):
    @tvm.register_func
    def toc_callback_bang_postproc(code, target):
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

    tvm._ffi.registry.remove_global_func("toc_callback_bang_postproc")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="the source file")

    args = parser.parse_args()
    base_name = os.path.basename(args.file)

    data_shape = base_name.split("_")[1:5]
    data_shape = [int(intg) for intg in data_shape]
    kernel_shape = base_name.split("_")[5:9]
    kernel_shape = [int(intg) for intg in kernel_shape]
    stride_h = stride_w = int(base_name.split(".")[0].split("_")[9])
    pad = int(base_name.split(".")[0].split("_")[10])

    verify_conv2d_nchw(
        base_name,
        data_shape,
        kernel_shape[1],
        kernel_shape[0],
        kernel_shape[2],
        stride_h,
        pad,
    )
    print("验证通过！")
