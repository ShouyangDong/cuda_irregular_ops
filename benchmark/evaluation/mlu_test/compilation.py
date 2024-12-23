import glob
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

from benchmark.utils import run_mlu_compilation as run_compilation


def compile_mlu_file(file_name):
    base_name = os.path.basename(file_name)
    so_name = base_name.replace("mlu", "so")
    so_name = os.path.join(os.getcwd(), "benchmark/data/mlu_code_test", so_name)

    # Read the MLU code
    with open(file_name, "r") as f:
        code = f.read()

    # Read the macro definitions
    with open(os.path.join(os.getcwd(), "benchmark/macro/mlu_macro.txt"), "r") as f:
        macro = f.read()

    # Combine macro with code
    code = macro + code
    back_file_name = file_name.replace(".mlu", "_bak.mlu")

    # Write the modified code into a new temporary file
    with open(back_file_name, mode="w") as f:
        f.write(code)

    # Run the compilation
    success, output = run_compilation(so_name, back_file_name)
    os.remove(back_file_name)  # Clean up the temporary backup file

    return success, output, so_name  # Return result and compiled shared object name


if __name__ == "__main__":
    files = glob.glob(os.path.join(os.getcwd(), "benchmark/data/mlu_code_test/*.mlu"))
    counter = 0

    # Use ProcessPoolExecutor for parallel compilation
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(compile_mlu_file, files), total=len(files)))

    # Process results
    for success, output, so_name in results:
        if success:
            counter += 1
            subprocess.run(["rm", so_name])  # Remove the compiled .so file
        else:
            print("[INFO] file_name: ", so_name)
            print(output)

    print(counter)
    print(len(files))
    print(
        "[INFO]*******************MLU Compilation success rate: ", counter / len(files)
    )
