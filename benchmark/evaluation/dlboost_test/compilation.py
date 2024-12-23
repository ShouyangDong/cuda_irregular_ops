import glob
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

from benchmark.utils import run_dlboost_compilation as run_compilation


def compile_cpp_file(file_name):
    base_name = os.path.basename(file_name)

    with open(file_name, "r") as f:
        code = f.read()

    with open("benchmark/macro/dlboost_macro.txt", "r") as f:
        macro = f.read()

    # Combine macro with code
    code = macro + code
    bak_file_name = file_name.replace(
        base_name.replace(".cpp", ""), base_name + "_bak.cpp"
    )

    # Write the modified code to a new temporary file
    with open(bak_file_name, mode="w") as f:
        f.write(code)

    so_name = base_name.replace("cpp", "so")
    so_name = os.path.join("benchmark/data/dlboost_code_test/", so_name)

    success, output = run_compilation(so_name, bak_file_name)
    os.remove(bak_file_name)  # Clean up temporary file

    return (
        success,
        output,
        so_name,
    )  # Return the result and the generated .so file name


if __name__ == "__main__":
    files = glob.glob("benchmark/data/dlboost_code_test/*.cpp")
    counter = 0

    # Using ProcessPoolExecutor for parallel compilation
    with ProcessPoolExecutor() as executor:
        results = list(
            tqdm(executor.map(compile_cpp_file, files), total=len(files))
        )

    # Process the results
    for success, output, so_name in results:
        if success:
            counter += 1
            subprocess.run(["rm", so_name])  # Remove the compiled .so file
        else:
            print(output)

    print(counter)
    print(len(files))
    print(
        "[INFO]*******************CPP Compilation success rate: ",
        counter / len(files),
    )
