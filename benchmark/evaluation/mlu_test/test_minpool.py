import argparse
import os

import numpy as np


def minpool_np(data, kernel_stride):
    """min pooling with numpy
    data : numpy.array
        input array

    kernel : list or tuple
        The kernel of avgpool

    stride : list or tuple
        The stride of avgpool
    """
    batch, dh, dw, dc = data.shape
    kh, kw, sh, sw = kernel_stride
    ch = (dh - kh) // sh + 1
    cw = (dw - kw) // sw + 1
    ret = np.zeros((batch, ch, cw, dc))
    for i in range(ch):
        for j in range(cw):
            mask = data[:, i * sh : i * sh + kh, j * sw : j * sw + kw, :]
            ret[:, i, j, :] = np.min(mask, axis=(1, 2))
    return ret


def generate_data(shape, dtype):
    return np.random.uniform(size=shape).astype(dtype)


def verify_pooling(name, file, shape, kernel, stride):
    op_name = name.split("_")[0]

    kh, kw = kernel[0], kernel[1]
    sh, sw = stride[0], stride[1]
    from toc import Environment

    env = Environment("cambricon/mlu590-h8")
    op_name = name.split("_")[0]

    @tvm.register_func("toc_callback_bang_postproc")
    def toc_callback_bang_postproc(code):

        if not os.path.exists(file):
            with open(file, "w", encoding="utf-8") as f:
                f.write(code)
        code = open(file, encoding="utf-8").read()
        code = code.replace("void " + op_name + "(", "void " + op_name + "_kernel0(")
        return code

    input0 = tsop.tensor(shape, dtype=bangpy.float32, name="input0")
    # Describ Computation
    result = tsop.minpool(input0, kh, kw, sh, sw)
    # Build ang get executable module
    fmlu = tsop.BuildBANG([input0], [result], "mlu590-h8", kernel_name=op_name)
    # Generate random test data and run on mlu and cpu

    data0 = generate_data(shape, "float32")
    cpu_output = minpool_np(data0, kernel_stride)
    result_np = np.zeros(shape=cpu_output.shape, dtype="float32")

    dev = bangpy.device(0)
    data_dev = bangpy.Array(data0, dev)
    result_arr = bangpy.Array(result_np, dev)

    fmlu(data_dev, result_arr)
    # Compare
    bangpy.assert_allclose(result_arr.numpy(), cpu_output, 0.1, 0)
    tvm._ffi.registry.remove_global_func("toc_callback_bang_postproc")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="the source file")
    args = parser.parse_args()
    base_name = os.path.basename(args.file)

    shape = base_name.split("_")[1:5]
    shape = [int(intg) for intg in shape]
    kernel_stride = base_name.split(".")[0].split("_")[5:]
    kernel_stride = [int(intg) for intg in kernel_stride]
    verify_pooling(base_name, file, shape, kernel_stride[:2], kernel_stride[2:])
    print("验证通过！")
