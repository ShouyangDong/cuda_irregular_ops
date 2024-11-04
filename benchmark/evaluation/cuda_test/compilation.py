import glob
import os
import subprocess

from tqdm import tqdm


def run_compilation(so_name, file_name):
    try:
        output = subprocess.run(
            ["nvcc", "-Xcompiler", "-fPIC", "-shared", "-o", so_name, file_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            check=True,
            text=True,
            timeout=15,
        )
        return True, output
    except subprocess.CalledProcessError as e:
        return False, e.output


files = glob.glob(os.path.join(os.getcwd(), "benchmark/data/cuda_code_test/*.cu"))
counter = 0
for file_name in tqdm(files):
    base_name = os.path.basename(file_name)
    so_name = base_name.replace("cu", "so")
    so_name = os.path.join(os.path.join(os.getcwd(), "benchmark/data/cuda_code_test/"), so_name)

    with open(file_name, "r") as f:
        code = f.read()

    with open(os.path.join(os.getcwd(), "benchmark/macro/cuda_macro.txt"), "r") as f:
        macro = f.read()

    code = macro + code
    back_file_name = file_name.replace(".cu", "_bak.cu")
    with open(back_file_name, mode="w") as f:
        f.write(code)
        f.close()

    success, output = run_compilation(so_name, back_file_name)
    os.remove(back_file_name)
    if success:
        counter += 1
        result = subprocess.run(["rm", so_name])
    else:
        print(output)

print(counter)
print(len(files))
print("[INFO]*******************Compilation rate: ", counter / len(files))
