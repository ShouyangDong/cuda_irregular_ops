import glob
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

from benchmark.utils import run_cuda_compilation as run_compilation


def compile_cuda_file(file_name):
    base_name = os.path.basename(file_name)
    so_name = base_name.replace("cu", "so")
    so_name = os.path.join(
        os.getcwd(), "benchmark/data/cuda_code_test/", so_name
    )

    with open(file_name, "r") as f:
        code = f.read()

    with open(
        os.path.join(os.getcwd(), "benchmark/macro/cuda_macro.txt"), "r"
    ) as f:
        macro = f.read()

    # Combine macro with code
    code = macro + code
    back_file_name = file_name.replace(".cu", "_bak.cu")

    # Write the combined code to a temporary file
    with open(back_file_name, mode="w") as f:
        f.write(code)

    success, output = run_compilation(so_name, back_file_name)
    os.remove(back_file_name)  # Clean up the temporary file

    return success, output


if __name__ == "__main__":
    files = glob.glob(
        os.path.join(os.getcwd(), "benchmark/data/cuda_code_test/*.cu")
    )
    counter = 0

    # Use ProcessPoolExecutor for parallel compilation
    with ProcessPoolExecutor() as executor:
        # Track the results of the compilation
        results = list(
            tqdm(executor.map(compile_cuda_file, files), total=len(files))
        )

    # Process results
    for success, output in results:
        if success:
            counter += 1
            so_name = (
                # Obtain the shared object name if needed (modify if necessary)
                output
            )
            subprocess.run(["rm", so_name])  # Remove the compiled .so file
        else:
            print(output)

    print(counter)
    print(len(files))
    print(
        "[INFO]*******************Compilation success rate: ",
        counter / len(files),
    )
