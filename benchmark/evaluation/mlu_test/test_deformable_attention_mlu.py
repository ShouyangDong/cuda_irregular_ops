import argparse
import os

import numpy as np
import toc
import torch
import tvm
import tvm.topi.testing
from toc import Environment
from tvm import te

env = Environment("cambricon/mlu590-h8")
import torch.nn.functional as F


@torch.no_grad()
def deformable_attention_pytorch(
    value, value_spatial_shapes, sampling_locations, attention_weights
):
    """Pytorch implementation of deformable attention from
    https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/functions/ms_deform_attn_func.py
    """
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = (
            value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
        )
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        N_ * M_, 1, Lq_, L_ * P_
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(N_, M_ * D_, Lq_)
    )
    return output.transpose(1, 2).contiguous()


def verify_deformable(name, file, shape):
    op_name = "deformable"
    N, M, D = shape[:3]
    Lq, L, P = shape[3:]
    shapes = torch.as_tensor(
        [[84, 117], [42, 59], [21, 30], [11, 15]], dtype=torch.long
    )
    S = sum([(H * W).item() for H, W in shapes])
    value = torch.rand(N, S, M, D) * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 2)
    attention_weights = torch.rand(N, Lq, M, L, P) + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)

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

    @tvm.register_func("toc_callback_bang_postproc")
    def toc_callback_bang_postproc(code):
        with open(file, "r") as f:
            code = f.read()
            f.close()
        code = code.replace("void " + op_name + "(", "void " + op_name + "_kernel0(")
        return code

    def test_deformable(A, B, C, D, E):
        n = A.shape[0]
        prod = np.prod(A.shape[:-1])
        ib = tvm.tir.ir_builder.create()
        tx = te.thread_axis("threadIdx.x")
        bx = te.thread_axis("blockIdx.x")
        ib.scope_attr(tx, "thread_extent", 4)
        ib.scope_attr(bx, "thread_extent", N)

        Aptr = ib.buffer_ptr(A)
        Bptr = ib.buffer_ptr(B)
        Cptr = ib.buffer_ptr(C)
        Dptr = ib.buffer_ptr(D)
        Eptr = ib.buffer_ptr(E)
        with ib.for_range(0, n, name="i") as i:
            Eptr[i] = Aptr[i] + Bptr[i] + Cptr[i] + Dptr[i] 
        body = ib.get()
        return body

    out_D = te.extern(
        output_shape,
        [A, shape_pl, B, C],
        lambda ins, outs: test_deformable(ins[0], ins[1], ins[2], ins[3], outs[0]),
        name=op_name,
        dtype="float32",
    )

    s = te.create_schedule(out_D.op)

    dev = tvm.device("bang", 0)
    a = tvm.nd.array(value, dev)
    b = tvm.nd.array(shapes.int(), dev)
    d = tvm.nd.array(sampling_locations, dev)
    e = tvm.nd.array(attention_weights, dev)
    output = tvm.nd.array(np.zeros(output_shape, "float32"), dev)
    with toc.build_config(env):
        f = toc.build(s, [A, shape_pl, B, C, out_D], name=op_name)
    f(a, b,  d, e, output)
    tvm._ffi.registry.remove_global_func("toc_callback_bang_postproc")
    torch_da = deformable_attention_pytorch(
        value, shapes, sampling_locations, attention_weights
    )
    np.testing.assert_allclose(
        output.numpy(),
        torch_da.numpy(),
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
    verify_deformable(base_name, args.file, shape)
    print("验证通过！")
