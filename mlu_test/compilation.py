import subprocess
import os
import glob


def run_compilation(so_name, file_name):
    try:
        output = subprocess.run(
            ["cncc", "-shared", "-fPIC", "--bang-mlu-arch=mtp_592", "-o", so_name, file_name],
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


files = glob.glob("./mlu_code_test/*.mlu")
counter = 0
for file_name in files:
    base_name = os.path.basename(file_name)
    so_name = base_name.replace("mlu", "so")
    so_name = os.path.join("./mlu_code_test/", so_name)

    with open(file_name, "r") as f:
        code = f.read()

    with open("./macro/mlu_macro.txt", "r") as f:
        macro = f.read()


    code = macro + code
    back_file_name = file_name.replace(".mlu", "_bak.mlu")
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
print("[INFO]*******************MLU Compilation rate: ", counter / len(files))