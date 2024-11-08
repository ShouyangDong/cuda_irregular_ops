import os
import subprocess

test_file_map = {
    "deformable": "benchmark/evaluation/{target}_test/test_deformable_attention.py",
    "layernorm": "benchmark/evaluation/{target}_test/test_layer_norm_cuda.py",
    "mha": "benchmark/evaluation/{target}_test/test_mha_cuda.py",
    "rmsnorm": "benchmark/evaluation/{target}_test/test_rms_norm_cuda.py",
    "gemm": "benchmark/evaluation/{target}_test/test_gemm.py",
    "gemv": "benchmark/evaluation/{target}_test/test_gemv.py",
    "bmm": "benchmark/evaluation/{target}_test/test_bmm.py",
    "conv1d": "benchmark/evaluation/{target}_test/test_conv1d.py",
    "conv2d": "benchmark/evaluation/{target}_test/test_conv2d.py",
    "conv2dnchw": "benchmark/evaluation/{target}_test/test_conv2d.py",
    "depthwiseconv": "benchmark/evaluation/{target}_test/test_depthwiseconv.py",
    "add": "benchmark/evaluation/{target}_test/test_add.py",
    "sign": "benchmark/evaluation/{target}_test/test_sign.py",
    "avgpool": "benchmark/evaluation/{target}_test/test_avgpool.py",
    "maxpool": "benchmark/evaluation/{target}_test/test_maxpool.py",
    "minpool": "benchmark/evaluation/{target}_test/test_minpool.py",
    "sumpool": "benchmark/evaluation/{target}_test/test_sumpool.py",
    "relu": "benchmark/evaluation/{target}_test/test_relu.py",
    "sigmoid": "benchmark/evaluation/{target}_test/test_sigmoid.py",
    "gelu": "benchmark/evaluation/{target}_test/test_gelu.py",
    "softmax": "benchmark/evaluation/{target}_test/test_softmax.py",
}


def run_test(file_name, test_file):
    try:
        output = subprocess.run(
            ["python", test_file, "--file", file_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            check=True,
            text=True,
            timeout=400,
        )
        return True, output
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except subprocess.CalledProcessError as e:
        return False, e.output


def unit_test(file_name, code):
    # 创建临时目录
    tmp_dir = "./tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    
    # 去掉扩展名
    filename_no_ext, _ = os.path.splitext(file_name)

    # 判断文件类型并设置目标
    if "__mlu_global" in code:
        target, file_type = "mlu", ".mlu"
    elif "__global__" in code:
        target, file_type = "cuda", ".cu"
    else:
        target, file_type = "cpp", ".cpp"

    # 生成目标文件名
    filename = filename_no_ext + file_type
    with open(os.path.join(tmp_dir, os.path.basename(filename)), mode="w") as f:
        f.write(code)

    # 提取操作名称，并生成测试文件路径
    op_name = os.path.basename(filename_no_ext).split("_")[0]
    test_file = test_file_map.get(op_name, "").format(target=target)

    # 运行测试
    success, output = run_test(file_name, test_file)
    return success
