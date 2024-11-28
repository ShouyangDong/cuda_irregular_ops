import argparse
import os

import numpy as np
import toc
import tvm
import tvm.topi.testing
from toc import Environment
from tvm import te

env = Environment("cambricon/mlu590-h8")


def get_im2col_indices(images_shape, filter_shape, padding, stride):
    """Get index for shape"""
    batch_size, channels, height, width = images_shape
    filter_height, filter_width = filter_shape
    pad_h, pad_w = padding
    stride_h, stride_w = stride
    out_height = int((height + np.sum(pad_h) - filter_height) / stride_h + 1)
    out_width = int((width + np.sum(pad_w) - filter_width) / stride_w + 1)

    i0 = np.repeat(np.arange(filter_height), filter_width)
    i0 = np.tile(i0, channels)
    i1 = stride_h * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(filter_width), filter_height * channels)
    j1 = stride_w * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(channels), filter_height * filter_width).reshape(-1, 1)
    return (k, i, j)


def image_to_column(images, filter_shape, stride, pad):
    """Transpose the input for conv"""
    filter_height, filter_width = filter_shape
    pad_h, pad_w = cpu_pad(pad)
    images_padded = np.pad(images, ((0, 0), (0, 0), pad_h, pad_w), mode="constant")
    k, i, j = get_im2col_indices(images.shape, filter_shape, (pad_h, pad_w), stride)
    cols = images_padded[:, k, i, j]
    channels = images.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(filter_height * filter_width * channels, -1)
    return cols


def cpu_pad(padding=None):
    """Get the pad param"""
    pad_h = 0
    pad_w = 0
    if isinstance(padding, (list, tuple)):
        if len(padding) == 2:
            pad_h = padding[0]
            pad_w = padding[1]
        elif len(padding) == 1:
            pad_h = pad_w = padding[0]
    elif isinstance(padding, int):
        pad_h = pad_w = padding
    return (pad_h, pad_h), (pad_w, pad_w)


def cpu_conv(data, kernel, stride_w, stride_h, pad=None):
    """Conv op in cpu"""
    batch_size, input_height, input_width, input_channel = data.shape
    output_channel, kernel_height, kernel_width, _ = kernel.shape
    pad_h, pad_w = cpu_pad(pad)
    out_height = int((input_height + np.sum(pad_h) - kernel_height) / stride_h + 1)
    out_width = int((input_width + np.sum(pad_w) - kernel_width) / stride_w + 1)
    data = data.transpose(0, 3, 1, 2)
    kernel = kernel.transpose(0, 3, 1, 2)
    X_col = image_to_column(
        data, (kernel_height, kernel_width), (stride_h, stride_w), pad
    )
    W_col = kernel.reshape((output_channel, -1))
    output = W_col.dot(X_col)
    output = output.reshape(output_channel, out_height, out_width, batch_size)
    return output.transpose(3, 1, 2, 0)


def verify_conv2d(name, file, shape, kernel, output_shape, stride, pad):
    op_name = "conv2d"

    @tvm.register_func("toc_callback_bang_postproc")
    def toc_callback_bang_postproc(code):
        with open(file, "r") as f:
            code = f.read()
            f.close()
        code = code.replace("void " + op_name + "(", "void " + op_name + "_kernel0(")
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
    tvm._ffi.registry.remove_global_func("toc_callback_bang_postproc")

    # cpu compute
    result_cpu = cpu_conv(data_np, kernel_np, stride, stride, pad)
    # Check if the results match
    np.testing.assert_allclose(
        result_dev.numpy(),
        result_cpu,
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
    data_shape = base_name.split("_")[1:5]
    data_shape = [int(intg) for intg in data_shape]

    kernel_shape = base_name.split("_")[5:9]
    kernel_shape = [int(intg) for intg in kernel_shape]
    stride_h = stride_w = int(base_name.split("_")[9])
    pad_h = pad_w = int(base_name.split("_")[10].replace(".mlu", ""))

    batch_size, input_height, input_width, input_channel = data_shape
    output_channel, kernel_height, kernel_width, _ = kernel_shape
    out_height = int((input_height + np.sum(pad_h) - kernel_height) / stride_h + 1)
    out_width = int((input_width + np.sum(pad_w) - kernel_width) / stride_w + 1)
    output_shape = [batch_size, out_height, out_width, output_channel]

    verify_conv2d(
        base_name, args.file, data_shape, kernel_shape, output_shape, stride_h, pad_h
    )
    print("验证通过！")
