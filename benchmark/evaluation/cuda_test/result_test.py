import glob
import os

from benchmark.utils import run_test

if __name__ == "__main__":
    files = glob.glob(
        os.path.join(os.getcwd(), "benchmark/data/cuda_code_test/**.cu")
    )
    counter = 0

    for file in files:
        base_name = os.path.basename(file)
        name = base_name.split("_")[0]
        if name == "deformable":
            success, output = run_test(
                file,
                "benchmark/evaluation/cuda_test/test_deformable_attention.py",
            )
        elif name == "layernorm":
            success, output = run_test(
                file, "benchmark/evaluation/cuda_test/test_layer_norm.py"
            )
        elif name == "mha":
            success, output = run_test(
                file, "benchmark/evaluation/cuda_test/test_mha.py"
            )
        elif name == "rmsnorm":
            success, output = run_test(
                file, "benchmark/evaluation/cuda_test/test_rms_norm.py"
            )
        elif name == "gemm":
            success, output = run_test(
                file, "benchmark/evaluation/cuda_test/test_gemm.py"
            )
        elif name == "gemv":
            success, output = run_test(
                file, "benchmark/evaluation/cuda_test/test_gemv.py"
            )
        elif name == "bmm":
            success, output = run_test(
                file, "benchmark/evaluation/cuda_test/test_bmm.py"
            )
        elif name == "conv1d":
            success, output = run_test(
                file, "benchmark/evaluation/cuda_test/test_conv1d.py"
            )
        elif name == "conv2d":
            success, output = run_test(
                file, "benchmark/evaluation/cuda_test/test_conv2d.py"
            )
        elif name == "conv2dnchw":
            success, output = run_test(
                file, "benchmark/evaluation/cuda_test/test_conv2dNCHW.py"
            )
        elif name == "depthwiseconv":
            success, output = run_test(
                file, "benchmark/evaluation/cuda_test/test_depthwiseconv.py"
            )
        elif name == "add":
            success, output = run_test(
                file, "benchmark/evaluation/cuda_test/test_add.py"
            )
        elif name == "sign":
            success, output = run_test(
                file, "benchmark/evaluation/cuda_test/test_sign.py"
            )
        elif name == "avgpool":
            success, output = run_test(
                file, "benchmark/evaluation/cuda_test/test_avgpool.py"
            )
        elif name == "maxpool":
            success, output = run_test(
                file, "benchmark/evaluation/cuda_test/test_maxpool.py"
            )
        elif name == "minpool":
            success, output = run_test(
                file, "benchmark/evaluation/cuda_test/test_minpool.py"
            )
        elif name == "sumpool":
            success, output = run_test(
                file, "benchmark/evaluation/cuda_test/test_sumpool.py"
            )
        elif name == "relu":
            success, output = run_test(
                file, "benchmark/evaluation/cuda_test/test_relu.py"
            )
        elif name == "sigmoid":
            success, output = run_test(
                file, "benchmark/evaluation/cuda_test/test_sigmoid.py"
            )
        elif name == "gelu":
            success, output = run_test(
                file, "benchmark/evaluation/cuda_test/test_gelu.py"
            )
        elif name == "softmax":
            success, output = run_test(
                file, "benchmark/evaluation/cuda_test/test_softmax.py"
            )
        else:
            raise RuntimeError("The file is not tested")

        if hasattr(output, "stdout") and "验证通过！" in output.stdout:
            counter += 1

        elif isinstance(output, str):
            print(base_name)
            print(output)

    print(counter)
    print(len(files))
    print(
        "[INFO]*******************cuda test computation successfule rate: ",
        counter / len(files),
    )
