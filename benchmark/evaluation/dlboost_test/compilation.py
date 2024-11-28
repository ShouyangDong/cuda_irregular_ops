import glob
import os
import subprocess

from tqdm import tqdm

from benchmark.utils import run_compilation

files = glob.glob("benchmark/data/dlboost_code_test/*.cpp")
counter = 0
for file_name in tqdm(files):
    base_name = os.path.basename(file_name)

    with open(file_name, "r") as f:
        code = f.read()
        f.close()

    with open("benchmark/macro/dlboost_macro.txt", "r") as f:
        macro = f.read()

    code = macro + code
    file_name = file_name.replace(base_name.replace(".cpp", ""), base_name + "_bak.cpp")
    with open(file_name, mode="w") as f:
        f.write(code)
        f.close()

    so_name = base_name.replace("cpp", "so")
    so_name = os.path.join("benchmark/data/dlboost_code_test/", so_name)

    success, output = run_compilation(so_name, file_name)
    os.remove(file_name)
    if success:
        counter += 1
        result = subprocess.run(["rm", so_name])
    else:
        print(output)


print(counter)
print(len(files))
print("[INFO]*******************CPP Compilation rate: ", counter / len(files))
