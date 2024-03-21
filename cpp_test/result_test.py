import subprocess
import glob
import os


def run_test(file_name, test_file, conversion=False):
    try:
        if conversion:
            output = subprocess.run(
                ["python", test_file, "--file", file_name, "--conversion"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding="utf-8",
                check=True,
                text=True,
                timeout=40,
            )
        else:
            output = subprocess.run(
                ["python", test_file, "--file", file_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding="utf-8",
                check=True,
                text=True,
                timeout=40,
            )           
        return True, output
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except subprocess.CalledProcessError as e:
        return False, e.output

if __name__ == "__main__":
    # cuda_2_cpp
    files = glob.glob("./cpp_code_test/*.cpp")
    counter = 0

    for file in files:
        base_name = os.path.basename(file)
        name = base_name.split("_")[0]
        if name == "deformable":
            success, output = run_test(file, "./test/cpp_kernel_test/test_unary.py", conversion=True)
        elif name == "layernorm":
            success, output = run_test(file, "./test/cpp_kernel_test/test_binary.py", conversion=True)
        elif name == "mha":
            success, output = run_test(file, "./test/cpp_kernel_test/test_pool.py", conversion=True)
        elif name == "rmsnorm":
            success, output = run_test(file, "./test/cpp_kernel_test/test_softmax.py", conversion=True)

        if hasattr(output, "stdout") and "验证通过" in output.stdout:
            counter += 1

        elif isinstance(output, str):
            print(base_name)
            print(output)

    print(counter)
    print(len(files))
    print("[INFO]*******************CUDA 2 CPP test successfule rate: ",  counter / len(files))