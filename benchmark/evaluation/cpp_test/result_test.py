import glob
import os
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

from benchmark.utils import run_test


def run_test_for_file(file):
    base_name = os.path.basename(file)
    name = base_name.split("_")[0]

    test_script_map = {
        "deformable": "test_deformable_attention.py",
        "layernorm": "test_layer_norm.py",
        "mha": "test_mha.py",
        "rmsnorm": "test_rms_norm.py",
        "gemm": "test_gemm.py",
        "gemv": "test_gemv.py",
        "bmm": "test_bmm.py",
        "conv1d": "test_conv1d.py",
        "conv2d": "test_conv2d.py",
        "conv2dnchw": "test_conv2dNCHW.py",
        "depthwiseconv": "test_depthwise_conv.py",
        "add": "test_add.py",
        "sign": "test_sign.py",
        "avgpool": "test_avgpool.py",
        "maxpool": "test_maxpool.py",
        "minpool": "test_minpool.py",
        "sumpool": "test_sumpool.py",
        "relu": "test_relu.py",
        "sigmoid": "test_sigmoid.py",
        "gelu": "test_gelu.py",
        "softmax": "test_softmax.py",
    }

    if name not in test_script_map:
        raise RuntimeError("This file is not tested.")

    test_script = os.path.join(
        os.getcwd(), "benchmark/evaluation/cpp_test", test_script_map[name]
    )
    success, output = run_test(file, test_script)

    return base_name, output


if __name__ == "__main__":
    files = glob.glob(os.path.join(os.getcwd(), "benchmark/data/cpp_code_test/*.cpp"))
    counter = 0

    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(run_test_for_file, files), total=len(files)))

    for base_name, output in results:
        if hasattr(output, "stdout") and "验证通过" in output.stdout:
            counter += 1
        elif isinstance(output, str):
            print(base_name)
            print(output)

    print(counter)
    print(len(files))
    print(
        "[INFO]*******************CPP test computation successful rate: ",
        counter / len(files),
    )
