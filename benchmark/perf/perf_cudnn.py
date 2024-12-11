import csv
import glob
import os
import timeit

import torch
import torch.nn.functional as F
# import MultiScaleDeformableAttention as MSDA
import shutil
torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.deterministic = True

device = torch.device("cuda")

def perf_elementwise(name, shape):
    x = torch.randn(shape, device=device)
    y = torch.randn(shape, device=device)

    op_name = name.split("_")[0]
    if op_name == "add":
        for _ in range(100): # warm up
            z = torch.add(x, y)

        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        nb_iters = 1000
        start.record()
        with torch.profiler.profile(activities=(torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA),
            record_shapes=True, on_trace_ready=torch.profiler.tensorboard_trace_handler("./time")) as p:
            for _ in range(nb_iters):
                z = torch.add(x, y)
        end.record()
        torch.cuda.current_stream().synchronize()
        print(p.key_averages().table(sort_by="self_cuda_time_total"))
        key_averages = p.key_averages()
        time = 0
        for avg in key_averages:
            if avg.key == 'aten::add':  # 根据 key 找到你需要的操作
                time = avg.cuda_time
        
        return time

    elif op_name == "sign":
        for _ in range(100): # warm up
            z = torch.sign(x)

        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        nb_iters = 1000
        start.record()
        with torch.profiler.profile(activities=(torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA),
            record_shapes=True, on_trace_ready=torch.profiler.tensorboard_trace_handler("./time")) as p:
            for _ in range(nb_iters):
                z = torch.sign(x)
        end.record()
        torch.cuda.current_stream().synchronize()
        print(p.key_averages().table(sort_by="self_cuda_time_total"))
        key_averages = p.key_averages()
        time = 0
        for avg in key_averages:
            if avg.key == 'aten::sign':  # 根据 key 找到你需要的操作
                time = avg.cuda_time
        
        return time


def perf_pooling(name, shape, kernel, stride):

    # 创建一个随机的输入张量
    x = torch.randn(shape, device=device)
    # 定义 MaxPooling 操作
    op_name = name.split("_")[0]
    name = "avg_pool2d"
    pool = torch.nn.AvgPool2d(kernel_size=kernel, stride=stride)
    if op_name == "maxpool":
        pool = torch.nn.MaxPool2d(kernel_size=kernel, stride=stride)
        name = "max_pool2d"
    elif op_name == "minpool":
        name = "avg_pool2d"
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

    for _ in range(100): # warm up
        test_pool()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    nb_iters = 1000
    start.record()        
    with torch.profiler.profile(activities=(torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA),
            record_shapes=True, on_trace_ready=torch.profiler.tensorboard_trace_handler("./time")) as p:
        for _ in range(nb_iters):
            test_pool()
    end.record()
    torch.cuda.current_stream().synchronize()

    print(p.key_averages().table(sort_by="self_cuda_time_total"))
    key_averages = p.key_averages()
    time = 0
    for avg in key_averages:
        if avg.key == ('aten::'+name):  # 根据 key 找到你需要的操作
            time = avg.cuda_time
    
    return time


def perf_bmm(name, shape_A, shape_B):

    # 创建随机张量
    A = torch.randn(shape_A, device=device)
    B = torch.randn(shape_B, device=device)

    def test_gemm():
        # 执行矩阵乘法操作 (GEMM)
        C = torch.matmul(A, B)
    for _ in range(100):
        test_gemm()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    nb_iters = 1000
    start.record()
    with torch.profiler.profile(activities=(torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA),
        record_shapes=True, on_trace_ready=torch.profiler.tensorboard_trace_handler("./time")) as p:
        for _ in range(nb_iters):
            test_gemm()
    end.record()
    torch.cuda.current_stream().synchronize()
    print(p.key_averages().table(sort_by="self_cuda_time_total"))
    key_averages = p.key_averages()
    time = 0
    for avg in key_averages:
        if avg.key == 'aten::matmul':  # 根据 key 找到你需要的操作
            time = avg.cuda_time
    
    return time


def perf_activation(name, shape):

    x = torch.randn(shape, device=device)
    op_name = name.split("_")[0]
    activation = torch.nn.ReLU()
    if op_name == "sigmoid":
        activation = torch.nn.Sigmoid()
    elif op_name == "gelu":
        activation = torch.nn.GELU()
    elif op_name == "softmax":
        activation = torch.nn.Softmax(dim=len(shape)-1)

    def test_activation():
        output = activation(x)

    for _ in range(100):
        test_activation()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    nb_iters = 1000
    start.record()
    with torch.profiler.profile(activities=(torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA),
        record_shapes=True, on_trace_ready=torch.profiler.tensorboard_trace_handler("./time")) as p:
        for _ in range(nb_iters):
            test_activation()
    end.record()
    torch.cuda.current_stream().synchronize()

    print(p.key_averages().table(sort_by="self_cuda_time_total"))
    key_averages = p.key_averages()
    time = 0
    for avg in key_averages:
        if avg.key == ('aten::'+ op_name):  # 根据 key 找到你需要的操作
            time = avg.cuda_time
    
    return time



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

    for _ in range(100):
        test_conv2d()
        
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    nb_iters = 1000
    start.record()
    with torch.profiler.profile(activities=(torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA),
        record_shapes=True, on_trace_ready=torch.profiler.tensorboard_trace_handler("./time")) as p:
        for _ in range(nb_iters):
            test_conv2d()
    end.record()
    torch.cuda.current_stream().synchronize()

    print(p.key_averages().table(sort_by="self_cuda_time_total"))
    key_averages = p.key_averages()
    time = 0
    for avg in key_averages:
        if avg.key == 'aten::conv2d':  # 根据 key 找到你需要的操作
            time = avg.cuda_time
    
    return time


def perf_conv2d_nhwc(name, shape, in_channels, out_channels, kernel, stride, padding):
    input_nhwc = torch.randn(shape, device=device)
    weight_hwio = torch.randn([out_channels, kernel, kernel, shape[3]], device=device)
    print(weight_hwio.shape)
    def test_conv2d():
        # 将输入从 NHWC 转换到 NCHW
        input_nchw = input_nhwc.permute(0, 3, 1, 2)
        
        # 将卷积核从 HWIO (H, W, in_channels, out_channels) 转换到 PyTorch的 OIHW 格式
        weight_oihw = weight_hwio.permute(0, 3, 1, 2)
        
        # 使用转换后的卷积核和输入进行卷积操作
        output_nchw = F.conv2d(input_nchw, weight_oihw,stride=stride, padding=padding)
        
        # 将输出从 NCHW 转换回 NHWC
        output_nhwc = output_nchw.permute(0, 3, 1, 2)
        

        
    for _ in range(100):
        test_conv2d()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    nb_iters = 1000
    start.record()
    with torch.profiler.profile(activities=(torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA),
        record_shapes=True, on_trace_ready=torch.profiler.tensorboard_trace_handler("./time")) as p:
        for _ in range(nb_iters):
            test_conv2d()
    end.record()
    torch.cuda.current_stream().synchronize()

    print(p.key_averages().table(sort_by="self_cuda_time_total"))
    key_averages = p.key_averages()
    time = 0
    for avg in key_averages:
        if avg.key == 'aten::conv2d':  # 根据 key 找到你需要的操作
            time = avg.cuda_time
    
    return time

def perf_gemv(name, shape):

    # 创建随机矩阵和向量 M, N = 1000, 1000
    matrix = torch.randn(shape, device=device)
    vector = torch.randn(shape[1], device=device)

    def test_gemv():
        output = torch.matmul(matrix, vector)
        # 或者使用 matrix @ vector
        
    for _ in range(100):
        test_gemv()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    nb_iters = 1000
    start.record()
    with torch.profiler.profile(activities=(torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA),
        record_shapes=True, on_trace_ready=torch.profiler.tensorboard_trace_handler("./time")) as p:
        for _ in range(nb_iters):
            test_gemv()
    end.record()
    torch.cuda.current_stream().synchronize()

    print(p.key_averages().table(sort_by="self_cuda_time_total"))
    key_averages = p.key_averages()
    time = 0
    for avg in key_averages:
        if avg.key == 'aten::matmul':  # 根据 key 找到你需要的操作
            time = avg.cuda_time
    
    return time


def perf_conv1d(name, shape):

    # 创建输入张量
    input_tensor = torch.randn([1 ,1, shape[1]], device=device)
    kernel_tensor = torch.randn([1, 1, 3], device=device)

    def test_conv1d():
        output_tensor = F.conv1d(input_tensor, kernel_tensor, padding=0)
        
    for _ in range(100):
        test_conv1d()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    nb_iters = 1000
    start.record()
    with torch.profiler.profile(activities=(torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA),
        record_shapes=True, on_trace_ready=torch.profiler.tensorboard_trace_handler("./time")) as p:
        for _ in range(nb_iters):
            test_conv1d()
    end.record()
    torch.cuda.current_stream().synchronize()

    print(p.key_averages().table(sort_by="self_cuda_time_total"))
    key_averages = p.key_averages()
    time = 0
    for avg in key_averages:
        if avg.key == 'aten::conv1d':  # 根据 key 找到你需要的操作
            time = avg.cuda_time
    
    return time


def perf_depthwise_conv2d(name, shape, kernel_size):
    input_hwio = torch.randn(shape, device=device)
    weight_fdio = torch.randn([kernel_size, kernel_size, shape[2]], device=device)

    def test_depthwise_conv2d():
        # 输入是 (height, width, in_depth)，添加一个批次维度变成 (1, height, width, in_depth)
        input_nchw = input_hwio.unsqueeze(0).permute(0, 3, 1, 2)  # 转换为 (1, in_depth, height, width)
        
        # 卷积核是 (fd, fd, in_depth)，需要转换为 (in_depth, 1, fd, fd)
        in_depth = weight_fdio.shape[2]
        weight_iodf = weight_fdio.permute(2, 0, 1).unsqueeze(1)  # 转换为 (in_depth, 1, fd, fd)
        
        # 使用深度可分离卷积
        output_nchw = F.conv2d(input_nchw, weight_iodf, groups=in_depth)
        
        # 转换输出格式从 (1, in_depth, new_height, new_width) 到 (new_height, new_width, in_depth)
        output_hwio = output_nchw.squeeze(0).permute(1, 2, 0)

        
    for _ in range(100):
        test_depthwise_conv2d()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    nb_iters = 1000
    start.record()
    with torch.profiler.profile(activities=(torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA),
        record_shapes=True, on_trace_ready=torch.profiler.tensorboard_trace_handler("./time")) as p:
        for _ in range(nb_iters):
            test_depthwise_conv2d()
    end.record()
    torch.cuda.current_stream().synchronize()

    print(p.key_averages().table(sort_by="self_cuda_time_total"))
    return start.elapsed_time(end) /nb_iters


def perf_layernorm(name, shape):

    # 创建输入张量
    input_tensor = torch.randn(shape, device=device)
    # 定义 LayerNorm 层
    layer_norm = torch.nn.LayerNorm(shape[-1], device=device)

    def test_layernorm():
        output = layer_norm(input_tensor)
    
    for _ in range(100):
        test_layernorm()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    nb_iters = 1000
    start.record()
    with torch.profiler.profile(activities=(torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA),
        record_shapes=True, on_trace_ready=torch.profiler.tensorboard_trace_handler("./time")) as p:
        for _ in range(nb_iters):
            test_layernorm()
    end.record()
    torch.cuda.current_stream().synchronize()

    print(p.key_averages().table(sort_by="self_cuda_time_total"))
    key_averages = p.key_averages()
    time = 0
    for avg in key_averages:
        if avg.key == 'aten::layer_norm':  # 根据 key 找到你需要的操作
            time = avg.cuda_time
    
    return time


def perf_rmsnorm(name, shape):

    # 创建输入张量
    input_tensor = torch.randn(shape, device=device)
    # 定义 RMSNorm 层
    rmsnorm = torch.nn.RMSNorm(shape, device=device)

    def test_rmsnorm():
        output = rmsnorm(input_tensor)

    for _ in range(100):
        test_rmsnorm()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    nb_iters = 1000
    start.record()
    with torch.profiler.profile(activities=(torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA),
        record_shapes=True, on_trace_ready=torch.profiler.tensorboard_trace_handler("./time")) as p:
        for _ in range(nb_iters):
            test_rmsnorm()
    end.record()
    torch.cuda.current_stream().synchronize()

    print(p.key_averages().table(sort_by="self_cuda_time_total"))
    key_averages = p.key_averages()
    time = 0
    for avg in key_averages:
        if avg.key == 'aten::rmsnorm':  # 根据 key 找到你需要的操作
            time = avg.cuda_time
    
    return time



def perf_deformable(name, shape):
    N, M, D = shape[:3]
    Lq, L, P = shape[3:]
    shapes = torch.as_tensor(
        [[84, 117], [42, 59], [21, 30], [11, 15]], dtype=torch.long, device=device
    )
    level_start_index = torch.cat(
        (shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1])
    )
    S = sum([(H * W).item() for H, W in shapes])

    value = torch.rand(N, S, M, D, device=device) * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 2, device=device)
    attention_weights = torch.rand(N, Lq, M, L, P, device=device) + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(
        -2, keepdim=True
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
        

    for _ in range(100):
        test_deformable()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    nb_iters = 1000
    start.record()
    with torch.profiler.profile(activities=(torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA),
        record_shapes=True, on_trace_ready=torch.profiler.tensorboard_trace_handler("./time")) as p:
        for _ in range(nb_iters):
            test_deformable()
    end.record()
    torch.cuda.current_stream().synchronize()

    print(p.key_averages().table(sort_by="self_cuda_time_total"))
    key_averages = p.key_averages()
    time = 0
    for avg in key_averages:
        if avg.key == 'aten::scaled_dot_product_attention':  # 根据 key 找到你需要的操作
            time = avg.cuda_time
    
    return time


def perf_scaled_dot_product_attention(name, shape):

    # Optionally use the context manager to ensure one of the fused kernels is run
    query = torch.rand(shape, dtype=torch.float16, device=device)
    key = torch.rand(shape, dtype=torch.float16, device=device)
    value = torch.rand(shape, dtype=torch.float16, device=device)

    def test_scaled_dot_product_attention():
        output = F.scaled_dot_product_attention(query, key, value)
        

    for _ in range(100):
        test_scaled_dot_product_attention()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    nb_iters = 1000
    start.record()
    with torch.profiler.profile(activities=(torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA),
        record_shapes=True, on_trace_ready=torch.profiler.tensorboard_trace_handler("./time")) as p:
        for _ in range(nb_iters):
            test_scaled_dot_product_attention()
    end.record()
    torch.cuda.current_stream().synchronize()

    print(p.key_averages().table(sort_by="self_cuda_time_total"))
    key_averages = p.key_averages()
    time = 0
    for avg in key_averages:
        if avg.key == 'aten::scaled_dot_product_attention':  # 根据 key 找到你需要的操作
            time = avg.cuda_time
    
    return time


if __name__ == "__main__":
    files = glob.glob(os.path.join(os.getcwd(), "benchmark/data/cuda_code_test/conv2dnchw*.cu"))
    counter = 0
    execution_time = 0
    table = []
    times = []
    for file in files:
        base_name = os.path.basename(file)
        name = base_name.split("_")[0]
        if name == "add" or name == "sign":
            shapes = base_name.split(".")[0]
            shape = [int(intg) for intg in shapes.split("_")[1:]]
            execution_time = perf_elementwise(base_name, shape)
            times.append(execution_time)

        elif name in ["avgpool", "maxpool", "minpool", "sumpool"]:
            shape = base_name.split("_")[1:5]
            shape = [int(intg) for intg in shape]
            kernel_stride = base_name.split(".")[0].split("_")[5:]
            kernel_stride = [int(intg) for intg in kernel_stride]
            execution_time = perf_pooling(
                base_name, shape, kernel_stride[0], kernel_stride[2]
            )
            times.append(execution_time)

        elif name == "bmm":
            shapes = base_name.split(".")[0]
            shape = [int(intg) for intg in shapes.split("_")[1:]]
            batch_size, matrix_dim_i, matrix_dim_j, matrix_dim_k = shape
            shape_A = [batch_size, matrix_dim_i, matrix_dim_j]
            shape_B = [batch_size, matrix_dim_j, matrix_dim_k]
            execution_time = perf_bmm(name, shape_A, shape_B)
            times.append(execution_time)

        elif name == "gemm":
            shapes = base_name.split(".")[0]
            shape = [int(intg) for intg in shapes.split("_")[1:]]
            shape_A = [shape[0], shape[1]]
            shape_B = [shape[1], shape[2]]
            execution_time = perf_bmm(name, shape_A, shape_B)
            times.append(execution_time)

        elif name in ["relu", "sigmoid", "gelu", "softmax"]:
            shapes = base_name.split(".")[0]
            shape = [int(intg) for intg in shapes.split("_")[1:]]
            execution_time = perf_activation(base_name, shape)
            times.append(execution_time)
        elif name == "conv2dnchw":
            data_shape = base_name.split("_")[1:5]
            data_shape = [int(intg) for intg in data_shape]
            kernel_shape = base_name.split("_")[5:9]
            kernel_shape = [int(intg) for intg in kernel_shape]
            stride_h = stride_w = int(base_name.split(".")[0].split("_")[9])
            pad = int(base_name.split(".")[0].split("_")[10])

            execution_time = perf_conv2d_nchw(
                base_name,
                data_shape,
                kernel_shape[1],
                kernel_shape[0],
                kernel_shape[2],
                stride_h,
                pad,
            )
            times.append(execution_time)
        elif name == "conv2d":
            data_shape = base_name.split("_")[1:5]
            data_shape = [int(intg) for intg in data_shape]
            kernel_shape = base_name.split("_")[5:9]
            kernel_shape = [int(intg) for intg in kernel_shape]
            stride_h = stride_w = int(base_name.split(".")[0].split("_")[9])
            pad = int(base_name.split(".")[0].split("_")[10])

            execution_time = perf_conv2d_nhwc(
                base_name,
                data_shape,
                kernel_shape[1],
                kernel_shape[0],
                kernel_shape[2],
                stride_h,
                pad,
            )
            times.append(execution_time)
        elif name == "gemv":
            shapes = base_name.split(".")[0]
            shape = [int(intg) for intg in shapes.split("_")[1:]]
            execution_time = perf_gemv(base_name, shape)
            times.append(execution_time)
        elif name == "conv1d":
            shapes = base_name.split(".")[0]
            shape = [int(intg) for intg in shapes.split("_")[1:]]
            execution_time = perf_conv1d(base_name, shape)
            times.append(execution_time)
        elif name == "depthwiseconv":
            shapes = base_name.split(".")[0]
            shape = [int(intg) for intg in shapes.split("_")[1:]]
            input_height, kernel_size, input_channels = shape[0], shape[1], shape[2]
            shape = [input_height, input_height, input_channels]
            execution_time = perf_depthwise_conv2d(base_name, shape, kernel_size)
            times.append(execution_time)
            
        elif name == "layernorm":
            shapes = base_name.split(".")[0]
            shape = [int(intg) for intg in shapes.split("_")[1:]]
            execution_time = perf_layernorm(base_name, shape)
            times.append(execution_time)

        elif name == "rmsnorm":
            #TODO:torch >= 2.4.0
            shapes = base_name.split(".")[0]
            shape = [int(intg) for intg in shapes.split("_")[1:]]
            execution_time = perf_rmsnorm(base_name, shape)
            # continue
            times.append(0)
        elif name == "deformable":
            # shapes = base_name.split(".")[0]
            # shape = [int(intg) for intg in shapes.split("_")[1:]]
            # perf_deformable(base_name, shape)
            # times.append(execution_time)
            # FIXME: incompatible pytorch version with RMSNorm
            continue
            times.append(0)

        elif name == "mha":
            shapes = base_name.split(".")[0]
            shape = [int(intg) for intg in shapes.split("_")[1:]]
            execution_time = perf_scaled_dot_product_attention(base_name, shape)
            times.append(execution_time)

    table.append(files)
    table.append(times)

    # 转置数据
    transposed_data = list(zip(*table))

    # 添加标题行
    header = ["file", "time(us)"]
    transposed_data.insert(0, header)

    # 保存为CSV文件
    with open("benchmark/perf/cudnn_output_maxpool.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(transposed_data)
    shutil.rmtree("./time")
