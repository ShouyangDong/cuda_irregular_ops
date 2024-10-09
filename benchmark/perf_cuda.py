import glob
import os
import tvm
import timeit
from tvm import te, topi
import tvm.topi.testing
import numpy as np
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


def perf_conv2d_nchw(name, shape, kernel, stride):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = torch.randn(16, 3, 224, 224, device=device)
    # 定义卷积层
    conv_layer = torch.nn.Conv2d(
        in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=0
    ).to(device)

    def test_conv2d():
        output = conv_layer(input_tensor)
        torch.cuda.synchronize()

    # 使用 timeit 进行多次测量，设置执行次数为 100
    execution_time = timeit.timeit(test_conv2d, number=100)
    print(f"{name} execution time: {execution_time * 10} ms")


def perf_conv2d_nchw(name, shape, in_channels, out_channels, kernel, stride, padding):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = torch.randn(shape, device=device)
    # 定义卷积层
    conv_layer = torch.nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel,
        stride=stride,
        padding=padding,
    ).to(device)

    def test_conv2d():
        output = conv_layer(input_tensor)
        torch.cuda.synchronize()

    # 使用 timeit 进行多次测量，设置执行次数为 100
    execution_time = timeit.timeit(test_conv2d, number=100)
    print(f"{name} execution time: {execution_time * 10} ms")


def perf_gemv(name, shape):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建随机矩阵和向量 M, N = 1000, 1000
    matrix = torch.randn(shape, device=device)
    vector = torch.randn(shape[1], device=device)

    def test_gemv():
        output = torch.matmul(matrix, vector)
        # 或者使用 matrix @ vector
        torch.cuda.synchronize()

    # 使用 timeit 进行多次测量，设置执行次数为 100
    execution_time = timeit.timeit(test_gemv, number=100)
    print(f"{name} execution time: {execution_time * 10} ms")


def perf_conv1d(name, shape):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建输入张量
    input_tensor = torch.randn(1, shape[1], device=device)
    # 定义卷积层
    conv_layer = torch.nn.Conv1d(
        in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0
    ).to(device)

    def test_conv1d():
        output = conv_layer(input_tensor)
        torch.cuda.synchronize()

    # 使用 timeit 进行多次测量，设置执行次数为 100
    execution_time = timeit.timeit(test_conv1d, number=100)
    print(f"{name} execution time: {execution_time * 10} ms")


def perf_depthwise_conv2d(name, shape):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建输入张量
    input_tensor = torch.randn(16, 3, 224, 224, device=device)
    # 定义深度卷积层
    depthwise_conv_layer = torch.nn.Conv2d(
        in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, groups=3
    ).to(device)

    def test_depthwise_conv2d():
        output = depthwise_conv_layer(input_tensor)
        torch.cuda.synchronize()

    # 使用 timeit 进行多次测量，设置执行次数为 100
    execution_time = timeit.timeit(test_depthwise_conv2d, number=100)
    print(f"{name} execution time: {execution_time * 10} ms")


def perf_layernorm(name, file, shape):
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

    def test_activation(A, B):
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


def perf_deformable(name, shape):
    N, M, D = shape[:3]
    Lq, L, P = shape[3:]
    shapes = torch.as_tensor(
        [[84, 117], [42, 59], [21, 30], [11, 15]], dtype=torch.long, device=device
    )
    level_start_index = torch.cat(
        (shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1])
    ).cuda()
    S = sum([(H * W).item() for H, W in shapes])

    value = torch.rand(N, S, M, D, device=device) * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 2, device=device)
    attention_weights = torch.rand(N, Lq, M, L, P, device=device) + 1e-5
    attention_weights /= (
        attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True).cuda()
    )

    def test_deformable():
        MSDA.ms_deform_attn_forward(
            value_pt,
            shapes_pt,
            value_level_start_index_pt,
            sampling_locations_pt,
            attention_weights_pt,
            64,
        )
        # necessary because kernel launches are async
        torch.cuda.synchronize()

    # 使用 timeit 进行多次测量，设置执行次数为 100
    execution_time = timeit.timeit(test_deformable, number=100)
    print(f"{name} execution time: {execution_time * 10} ms")


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
    files = glob.glob("./cuda_code_test/mha*.cu")
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

        elif name == "conv2dnchw":
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
            perf_gemv(base_name, shape)

        elif name == "conv1d":
            shapes = base_name.split(".")[0]
            shape = [int(intg) for intg in shapes.split("_")[1:]]
            perf_conv1d(base_name, shape)

        elif name == "depthwiseconv_1":
            shapes = base_name.split(".")[0]
            shape = [int(intg) for intg in shapes.split("_")[1:]]
            input_height, kernel_size, input_channels = shape[0], shape[1], shape[2]
            perf_depthwise_conv2d(base_name, shape)

        elif name == "layernorm":
            shapes = base_name.split(".")[0]
            shape = [int(intg) for intg in shapes.split("_")[1:]]
            perf_layernorm(base_name, file, shape)

        elif name == "rmsnorm":
            shapes = base_name.split(".")[0]
            shape = [int(intg) for intg in shapes.split("_")[1:]]
            perf_rmsnorm(base_name, file, shape)

        elif name == "deformable":
            # shapes = base_name.split(".")[0]
            # shape = [int(intg) for intg in shapes.split("_")[1:]]
            # perf_deformable(base_name, shape)
            # FIXME: incompatible pytorch version with RMSNorm
            continue

        elif name == "mha":
            shapes = base_name.split(".")[0]
            shape = [int(intg) for intg in shapes.split("_")[1:]]
            perf_scaled_dot_product_attention(base_name, file, shape)
