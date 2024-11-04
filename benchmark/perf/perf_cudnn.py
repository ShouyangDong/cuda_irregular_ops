import glob
import os
import timeit

import torch
import torch.nn.functional as F


def perf_elementwise(name, shape):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.randn(shape, device=device)
    y = torch.randn(shape, device=device)

    def test_add():
        z = torch.add(x, y)
        torch.cuda.synchronize()

    def test_sign():
        z = torch.sign(x)
        torch.cuda.synchronize()

    op_name = name.split("_")[0]
    if op_name == "add":
        # 使用 timeit 进行多次测量，设置执行次数为 100
        execution_time = timeit.timeit(test_add, number=100)
        print(f"{name} execution time: {execution_time * 10} ms")

    elif op_name == "sign":
        execution_time = timeit.timeit(test_sign, number=100)
        print(f"{name} execution time: {execution_time * 10} ms")


def perf_pooling(name, shape, kernel, stride):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建一个随机的输入张量
    x = torch.randn(shape, device=device)
    # 定义 MaxPooling 操作
    op_name = name.split("_")[0]

    pool = torch.nn.AvgPool2d(kernel_size=kernel, stride=stride)
    if op_name == "maxpool":
        pool = torch.nn.MaxPool2d(kernel_size=kernel, stride=stride)

    elif op_name == "minpool":

        class MinPool2d(torch.nn.Module):
            def __init__(self, kernel_size, stride=None, padding=0):
                super(MinPool2d, self).__init__()
                self.kernel_size = kernel_size
                self.stride = stride
                self.padding = padding

            def forward(self, x):
                # 取反输入
                x_neg = -x
                # 执行最大池化
                x_maxpool = F.max_pool2d(
                    x_neg, self.kernel_size, stride=self.stride, padding=self.padding
                )
                # 再取反结果
                return -x_maxpool

        # 使用自定义的 MinPool2d
        pool = MinPool2d(kernel_size=kernel, stride=stride)

    elif op_name == "sumpool":

        class SumPool2d(torch.nn.Module):
            def __init__(self, kernel_size, stride=None, padding=0):
                super(SumPool2d, self).__init__()
                self.kernel_size = kernel_size
                self.stride = stride
                self.padding = padding

            def forward(self, x):
                # 使用平均池化
                x_avgpool = F.avg_pool2d(
                    x,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                )
                # 乘以池化窗口的大小，获得求和池化效果
                return x_avgpool * (self.kernel_size**2)

        # 使用自定义的SumPool2d
        pool = SumPool2d(kernel_size=kernel, stride=stride)

    def test_pool():
        output = pool(x)
        torch.cuda.synchronize()

    # 使用 timeit 进行多次测量，设置执行次数为 100
    execution_time = timeit.timeit(test_pool, number=100)
    print(f"{name} execution time: {execution_time * 10} ms")


def perf_bmm(name, shape_A, shape_B):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建随机张量
    A = torch.randn(shape_A, device=device)
    B = torch.randn(shape_B, device=device)

    def test_gemm():
        # 执行矩阵乘法操作 (GEMM)
        C = torch.matmul(A, B)
        # 确保 CUDA 操作完成
        torch.cuda.synchronize()

    # 使用 timeit 进行多次测量，设置执行次数为 100
    execution_time = timeit.timeit(test_gemm, number=100)
    print(f"{name} execution time: {execution_time * 10} ms")


def perf_activation(name, shape):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(shape, device=device)
    op_name = name.split("_")[0]

    activation = torch.nn.ReLU()
    if name == "sigmoid":
        activation = torch.nn.Sigmoid()
    elif name == "gelu":
        activation = torch.nn.GELU()
    elif name == "softmax":
        activation = torch.nn.Softmax(dim=len(shape))

    def test_activation():
        output = activation(x)
        torch.cuda.synchronize()

    # 使用 timeit 进行多次测量，设置执行次数为 100
    execution_time = timeit.timeit(test_activation, number=100)
    print(f"{name} execution time: {execution_time * 10} ms")


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


def perf_layernorm(name, shape):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建输入张量
    input_tensor = torch.randn(shape, device=device)
    # 定义 LayerNorm 层
    layer_norm = torch.nn.LayerNorm(shape[-1]).to(device)

    def test_layernorm():
        output = layer_norm(input_tensor)
        torch.cuda.synchronize()

    # 使用 timeit 进行多次测量，设置执行次数为 100
    execution_time = timeit.timeit(test_layernorm, number=100)
    print(f"{name} execution time: {execution_time * 10} ms")


def perf_rmsnorm(name, shape):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建输入张量
    input_tensor = torch.randn(shape, device=device)
    # 定义 RMSNorm 层
    rmsnorm = torch.nn.RMSNorm(shape).to(device)

    def test_rmsnorm():
        output = rmsnorm(input_tensor)
        torch.cuda.synchronize()

    # 使用 timeit 进行多次测量，设置执行次数为 100
    execution_time = timeit.timeit(test_rmsnorm, number=100)
    print(f"{name} execution time: {execution_time * 10} ms")


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


def perf_scaled_dot_product_attention(name, shape):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Optionally use the context manager to ensure one of the fused kernels is run
    query = torch.rand(shape, device="cuda")
    key = torch.rand(shape, device="cuda")
    value = torch.rand(shape, device="cuda")

    def test_scaled_dot_product_attention():
        output = F.scaled_dot_product_attention(query, key, value)
        torch.cuda.synchronize()

    # 使用 timeit 进行多次测量，设置执行次数为 100
    execution_time = timeit.timeit(test_scaled_dot_product_attention, number=100)
    print(f"{name} execution time: {execution_time * 10} ms")


if __name__ == "__main__":
    files = glob.glob(os.path.join(os.getcwd(), "benchmark/data/cuda_code_test/*.cu"))
    counter = 0

    for file in files:
        base_name = os.path.basename(file)
        name = base_name.split("_")[0]
        if name == "add" or name == "sign":
            shapes = base_name.split(".")[0]
            shape = [int(intg) for intg in shapes.split("_")[1:]]
            perf_elementwise(base_name, shape)

        elif name in ["avgpool", "maxpool", "minpool", "sumpool"]:
            shape = base_name.split("_")[1:5]
            shape = [int(intg) for intg in shape]
            kernel_stride = base_name.split(".")[0].split("_")[5:]
            kernel_stride = [int(intg) for intg in kernel_stride]
            perf_pooling(base_name, shape, kernel_stride[0], kernel_stride[2])

        elif name == "bmm":
            shapes = base_name.split(".")[0]
            shape = [int(intg) for intg in shapes.split("_")[1:]]
            batch_size, matrix_dim_i, matrix_dim_j, matrix_dim_k = shape
            shape_A = [batch_size, matrix_dim_i, matrix_dim_j]
            shape_B = [batch_size, matrix_dim_j, matrix_dim_k]
            perf_bmm(name, shape_A, shape_B)

        elif name == "gemm":
            shapes = base_name.split(".")[0]
            shape = [int(intg) for intg in shapes.split("_")[1:]]
            shape_A = [shape[0], shape[1]]
            shape_B = [shape[1], shape[2]]
            perf_bmm(name, shape_A, shape_B)

        elif name in ["relu", "sigmoid", "gelu", "softmax"]:
            shapes = base_name.split(".")[0]
            shape = [int(intg) for intg in shapes.split("_")[1:]]
            perf_activation(base_name, shape)

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
            perf_layernorm(base_name, shape)

        elif name == "rmsnorm":
            shapes = base_name.split(".")[0]
            shape = [int(intg) for intg in shapes.split("_")[1:]]
            perf_rmsnorm(base_name, shape)

        elif name == "deformable":
            # shapes = base_name.split(".")[0]
            # shape = [int(intg) for intg in shapes.split("_")[1:]]
            # perf_deformable(base_name, shape)
            # FIXME: incompatible pytorch version with RMSNorm
            continue

        elif name == "mha":
            shapes = base_name.split(".")[0]
            shape = [int(intg) for intg in shapes.split("_")[1:]]
            perf_scaled_dot_product_attention(base_name, shape)
