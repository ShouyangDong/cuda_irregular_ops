import glob
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from benchmark.utils import run_test


def run_test_for_file(file):
    base_name = os.path.basename(file)
    name = base_name.split("_")[0]
    test_script_map = {
        "deformable": "benchmark/evaluation/cuda_test/test_deformable_attention.py",
        "layernorm": "benchmark/evaluation/cuda_test/test_layer_norm.py",
        "mha": "benchmark/evaluation/cuda_test/test_mha.py",
        "rmsnorm": "benchmark/evaluation/cuda_test/test_rms_norm.py",
        "gemm": "benchmark/evaluation/cuda_test/test_gemm.py",
        "gemv": "benchmark/evaluation/cuda_test/test_gemv.py",
        "bmm": "benchmark/evaluation/cuda_test/test_bmm.py",
        "conv1d": "benchmark/evaluation/cuda_test/test_conv1d.py",
        "conv2d": "benchmark/evaluation/cuda_test/test_conv2d.py",
        "conv2dnchw": "benchmark/evaluation/cuda_test/test_conv2dNCHW.py",
        "depthwiseconv": "benchmark/evaluation/cuda_test/test_depthwiseconv.py",
        "add": "benchmark/evaluation/cuda_test/test_add.py",
        "sign": "benchmark/evaluation/cuda_test/test_sign.py",
        "avgpool": "benchmark/evaluation/cuda_test/test_avgpool.py",
        "maxpool": "benchmark/evaluation/cuda_test/test_maxpool.py",
        "minpool": "benchmark/evaluation/cuda_test/test_minpool.py",
        "sumpool": "benchmark/evaluation/cuda_test/test_sumpool.py",
        "relu": "benchmark/evaluation/cuda_test/test_relu.py",
        "sigmoid": "benchmark/evaluation/cuda_test/test_sigmoid.py",
        "gelu": "benchmark/evaluation/cuda_test/test_gelu.py",
        "softmax": "benchmark/evaluation/cuda_test/test_softmax.py",
    }

    if name not in test_script_map:
        raise RuntimeError(f"The file {base_name} is not tested")

    test_script = test_script_map[name]
    success, output = run_test(file, test_script)
    return base_name, success, output


if __name__ == "__main__":
    files = glob.glob(
        os.path.join(os.getcwd(), "benchmark/data/cuda_code_test/*.cu")
    )
    counter = 0

    # Using tqdm to create a progress bar
    with ThreadPoolExecutor() as executor:
        future_to_file = {
            executor.submit(run_test_for_file, file): file for file in files
        }

        # Initialize a tqdm progress bar
        with tqdm(total=len(future_to_file), desc="Running Tests") as pbar:
            for future in as_completed(future_to_file):
                pbar.update(1)  # Update progress bar for each completed task
                file = future_to_file[future]
                try:
                    base_name, success, output = future.result()
                    if (
                        hasattr(output, "stdout")
                        and "验证通过！" in output.stdout
                    ):
                        counter += 1
                    elif isinstance(output, str):
                        print(base_name)
                        print(output)
                except Exception as exc:
                    print(f"{file} generated an exception: {exc}")

    print(counter)
    print(len(files))
    print(
        "[INFO]*******************cuda test computation successful rate: ",
        counter / len(files),
    )
