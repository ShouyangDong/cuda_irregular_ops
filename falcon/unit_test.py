import os
import subprocess

test_file_map = {
    "deformable": f"benchmark/evaluation/{taregt}_test/test_deformable_attention.py",
    "layernorm": f"benchmark/evaluation/{taregt}_test/test_layer_norm_cuda.py",
    "mha": f"benchmark/evaluation/{taregt}_test/test_mha_cuda.py",
    "rmsnorm": f"benchmark/evaluation/{taregt}_test/test_rms_norm_cuda.py",
    "gemm": f"benchmark/evaluation/{taregt}_test/test_gemm.py",
    "gemv": f"benchmark/evaluation/{taregt}_test/test_gemv.py",
    "bmm": f"benchmark/evaluation/{taregt}_test/test_bmm.py",
    "conv1d": f"benchmark/evaluation/{taregt}_test/test_conv1d.py",
    "conv2d": f"benchmark/evaluation/{taregt}_test/test_conv2d.py",
    "conv2dnchw": f"benchmark/evaluation/{taregt}_test/test_conv2d.py",
    "depthwiseconv": f"benchmark/evaluation/{taregt}_test/test_depthwiseconv.py",
    "add": f"benchmark/evaluation/{taregt}_test/test_add.py",
    "sign": f"benchmark/evaluation/{taregt}_test/test_sign.py",
    "avgpool": f"benchmark/evaluation/{taregt}_test/test_avgpool.py",
    "maxpool": f"benchmark/evaluation/{taregt}_test/test_maxpool.py",
    "minpool": f"benchmark/evaluation/{taregt}_test/test_minpool.py",
    "sumpool": f"benchmark/evaluation/{taregt}_test/test_sumpool.py",
    "relu": f"benchmark/evaluation/{taregt}_test/test_relu.py",
    "sigmoid": f"benchmark/evaluation/{taregt}_test/test_sigmoid.py",
    "gelu": f"benchmark/evaluation/{taregt}_test/test_gelu.py",
    "softmax": f"benchmark/evaluation/{taregt}_test/test_softmax.py",
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
    with open(file_name, mode="w") as f:
        f.write(code)
        f.close()
    target == "cpp"
    if "__mlu_global" in code:
        target = "mlu"
    elif "__global__" in code:
        target == "cuda"

    op_name = os.path.basename(file_name).split("_")[0]
    test_file = test_file_map[op_name].format(target)
    success, output = run_test(file_name, test_file)
    return success
