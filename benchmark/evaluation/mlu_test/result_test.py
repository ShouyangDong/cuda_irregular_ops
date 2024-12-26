import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from benchmark.utils import run_test


def process_file(file):
    base_name = os.path.basename(file)
    name = base_name.split("_")[0]
    test_file_mapping = {
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
        "depthwiseconv": "test_depthwiseconv.py",
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

    if name in test_file_mapping:
        test_file = os.path.join(
            os.getcwd(),
            "benchmark/evaluation/mlu_test",
            test_file_mapping[name],
        )
        success, output = run_test(file, test_file)
        return file, output
    return file, None


if __name__ == "__main__":
    files = glob.glob(
        os.path.join(os.getcwd(), "benchmark/data/mlu_code_test/bmm*.mlu")
    )
    counter = 0

    with ProcessPoolExecutor() as executor:
        future_to_file = {
            executor.submit(process_file, file): file for file in files
        }

        for future in tqdm(as_completed(future_to_file), total=len(files)):
            file, output = future.result()
            base_name = os.path.basename(file)

            if hasattr(output, "stdout") and "验证通过！" in output.stdout:
                counter += 1
            elif isinstance(output, str):
                print(base_name)
                print(output)

    print(f"Successful tests: {counter}")
    print(f"Total files: {len(files)}")
    print(
        f"[INFO]*******************MLU test computation successful rate: {counter / len(files):.2f}"
    )
