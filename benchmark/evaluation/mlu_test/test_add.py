import argparse
import os

import bangpy
import numpy as np
import tvm
import tvm.topi.testing
from bangpy import tensor_op as tsop


def verify_add(name, file, shape):
    from toc import Environment

    env = Environment("cambricon/mlu590-h8")
    op_name = name.split("_")[0]

    @tvm.register_func("toc_callback_bang_postproc")
    def toc_callback_bang_postproc(code):
        tvm._ffi.registry.remove_global_func("toc_callback_bang_postproc")
        if not os.path.exists(file):
            with open(file, "w", encoding="utf-8") as f:
                f.write(code)
        code = open(file, encoding="utf-8").read()
        code = code.replace("void " + op_name + "(", "void " + op_name + "_kernel0(")
        return code

    input0 = tsop.tensor(shape, dtype=bangpy.float32, name="input0")
    input1 = tsop.tensor(shape, dtype=bangpy.float32, name="input1")
    # Describe Computation
    result = tsop.add(input0, input1)
    # Build and get executable module
    fmlu = tsop.BuildBANG([input0, input1], [result], "mlu590-h8", kernel_name=op_name)
    # Generate random test data and run on mlu and cpu
    data_lhs = np.random.rand(*shape).astype("float32")
    data_rhs = np.random.rand(*shape).astype("float32")
    result_np = np.zeros(shape=shape, dtype="float32")
    dev = bangpy.device(0)
    data_lhs_dev = bangpy.Array(data_lhs, dev)
    data_rhs_dev = bangpy.Array(data_rhs, dev)
    result_arr = bangpy.Array(result_np, dev)

    fmlu(data_lhs_dev, data_rhs_dev, result_arr)
    mlu_output = result_arr
    cpu_output = np.add(data_lhs, data_rhs)
    bangpy.assert_allclose(mlu_output.numpy(), cpu_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="the source file")
    args = parser.parse_args()
    base_name = os.path.basename(args.file)
    shapes = base_name.split(".")[0]
    shape = [int(intg) for intg in shapes.split("_")[1:]]
    verify_add(base_name, args.file, shape)
    print("验证通过！")
