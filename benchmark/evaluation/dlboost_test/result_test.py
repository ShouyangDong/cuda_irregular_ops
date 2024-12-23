import glob
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from benchmark.utils import run_test


# 定义一个函数来处理每个文件的测试
def process_file(file):
    base_name = os.path.basename(file)
    name = base_name.split("_")[0]

    test_files = {
        "deformable": "benchmark/evaluation/dlboost_test/test_deformable_attention.py",
        "layernorm": "benchmark/evaluation/dlboost_test/test_layer_norm.py",
        "mha": "benchmark/evaluation/dlboost_test/test_mha.py",
        "rmsnorm": "benchmark/evaluation/dlboost_test/test_rms_norm.py",
        "gemm": "benchmark/evaluation/dlboost_test/test_gemm.py",
        "gemv": "benchmark/evaluation/dlboost_test/test_gemv.py",
        "bmm": "benchmark/evaluation/dlboost_test/test_bmm.py",
        "conv1d": "benchmark/evaluation/dlboost_test/test_conv1d.py",
        "conv2d": "benchmark/evaluation/dlboost_test/test_conv2d.py",
        "conv2dnchw": "benchmark/evaluation/dlboost_test/test_conv2dNCHW.py",
        "depthwiseconv": "benchmark/evaluation/dlboost_test/test_depthwise_conv.py",
        "add": "benchmark/evaluation/dlboost_test/test_add.py",
        "sign": "benchmark/evaluation/dlboost_test/test_sign.py",
        "avgpool": "benchmark/evaluation/dlboost_test/test_avgpool.py",
        "maxpool": "benchmark/evaluation/dlboost_test/test_maxpool.py",
        "minpool": "benchmark/evaluation/dlboost_test/test_minpool.py",
        "sumpool": "benchmark/evaluation/dlboost_test/test_sumpool.py",
        "relu": "benchmark/evaluation/dlboost_test/test_relu.py",
        "sigmoid": "benchmark/evaluation/dlboost_test/test_sigmoid.py",
        "gelu": "benchmark/evaluation/dlboost_test/test_gelu.py",
        "softmax": "benchmark/evaluation/dlboost_test/test_softmax.py",
    }

    if name in test_files:
        test_script = test_files[name]
        success, output = run_test(file, test_script)
        return base_name, success, output

    return base_name, False, f"No test script found for {name}"


# 主程序
if __name__ == "__main__":
    files = glob.glob("benchmark/data/dlboost_code_test/*.cpp")
    counter = 0
    results = []

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_file, file): file for file in files}

        for future in tqdm(as_completed(futures), total=len(files)):
            base_name, success, output = future.result()
            results.append((base_name, success, output))

            if (
                success
                and hasattr(output, "stdout")
                and "验证通过！" in output.stdout
            ):
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
