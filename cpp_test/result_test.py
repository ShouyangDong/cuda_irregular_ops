import subprocess
import glob
import os
from tqdm import tqdm

def run_test(file_name, test_file):
    try:
        output = subprocess.run(
            ["python", test_file, "--file", file_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            check=True,
            # text=True,
            timeout=400,
        )           
        return True, output
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except subprocess.CalledProcessError as e:
        return False, e.output

if __name__ == "__main__":
    files = glob.glob("./cpp_code_test/*.cpp")
    counter = 0

    for file in tqdm(files):
        base_name = os.path.basename(file)
        name = base_name.split("_")[0]
        if name == "deformable":
            success, output = run_test(file, "./cpp_test/test_deformable_attention_cpp.py")
        elif name == "layernorm":
            success, output = run_test(file, "./cpp_test/test_layer_norm_cpp.py")
        elif name == "mha":
            success, output = run_test(file, "./cpp_test/test_mha.py")
        elif name == "rmsnorm":
            success, output = run_test(file, "./cpp_test/test_rms_norm_cpp.py")
        elif name == "bmm":
            success, output = run_test(file, "./cpp_test/test_bmm.py")
        elif name == "conv1d":
            success, output = run_test(file, "./cpp_test/test_conv1d.py")
        elif name == "gemv":
            success, output = run_test(file, "./cpp_test/test_gemv.py")
        elif name == "depthwiseconv":
            success, output = run_test(file, "./cpp_test/test_depthwise_conv.py")

        if hasattr(output, "stdout") and "验证通过" in output.stdout:
            counter += 1

        elif isinstance(output, str):
            print(base_name)
            print(output)

    print(counter)
    print(len(files))
    print("[INFO]*******************CPP test successfule rate: ",  counter / len(files))