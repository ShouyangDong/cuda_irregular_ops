import glob
import os
import subprocess

import openai

openai.api_key = "sk-JmlwEmWiNtFqSD7IDaF981Dd8a7447FfBcE768755cB38010"
openai.api_base = "https://api.keya.pw/v1"


question_system_prompt = """You are an expert in the field of coding, and here is a code understanding task.
                            You will generate the corresponding code based on the hints provided.
                            You should only output the C function without any explanation and natural language.
                            Wrap your code with "```"
                            """


def run_test(file_name, test_file, conversion=False):
    try:
        if conversion:
            output = subprocess.run(
                ["python", test_file, "--file", file_name, "--conversion"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding="utf-8",
                check=True,
                # text=True,
                timeout=40,
            )
        else:
            output = subprocess.run(
                ["python", test_file, "--file", file_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding="utf-8",
                check=True,
                # text=True,
                timeout=40,
            )
        return True, output
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except subprocess.CalledProcessError as e:
        return False, e.output


unary_op = [
    "abs",
    "cos",
    "exp",
    "exp2",
    "gelu",
    "log",
    "reciprocal",
    "relu",
    "rsqrt",
    "sigmoid",
    "sign",
    "sin",
    "sqrt",
    "square",
    "tanh",
]
binary_op = [
    "add",
    "multiply",
    "subtract",
    "divide",
    "logicaland",
    "equal",
    "notequal",
    "greater",
    "greaterequal",
    "less",
    "lessequal",
    "maximum",
    "minimum",
]
pool_op = ["maxpool", "minpool", "avgpool", "sumpool"]


def run_C_test_file(file):
    base_name = os.path.basename(file)
    name = base_name.split("_")[0]
    if name in unary_op:
        success, output = run_test(
            file, "./test/cpp_kernel_test/test_unary.py", conversion=True
        )
    elif name in binary_op:
        success, output = run_test(
            file, "./test/cpp_kernel_test/test_binary.py", conversion=True
        )
    elif name in pool_op:
        success, output = run_test(
            file, "./test/cpp_kernel_test/test_pool.py", conversion=True
        )
    elif name == "conv":
        success, output = run_test(
            file, "./test/cpp_kernel_test/test_conv.py", conversion=True
        )
    elif name == "softmax":
        success, output = run_test(
            file, "./test/cpp_kernel_test/test_softmax.py", conversion=True
        )
    return success, output


def fix_compilation_code(code, error_msg):
    prompt = f"""You are a genius programmer that helps fix Code. You reflect on every mistake that you make and learn from it.
    The code provided contains syntax errors that cause compilation to fail.
    Please fix the syntax errors in the code based on the error messages during compilation. The function must start with "extern "C" void".
    C code:\n{code}\n
    error messag:\n{error_msg}\n

    Please provide only the complete fixed C code without any additional text or explanations.
    Please make sure that you don't add any other text, just post back the code.
    It is very important that you do that, because otherwise you will interfere with a very important task of mine."""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": question_system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return response["choices"][0]["message"]["content"]


def fix_computation_code(error_code, error_output):
    prompt = f""".The error code is originally translated from C or CUDA with the same functionality.
    But the error code has some bugs and I cannot find them. Help me correct the error code. Return
    the fixed error code.

    error code:\n{error_code}\n
    error messag:\n{error_output}\n

    Please provide only the complete fixed C code without any additional text or explanations.
    Please make sure that you don't add any other text, just post back the code.
    It is very important that you do that, because otherwise you will interfere with a very important task of mine."""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": question_system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return response["choices"][0]["message"]["content"]


def extract_C_code(string):
    if "```" in string:
        string = string.replace("``` cpp", "```")
        string = string.replace("```cpp", "```")
        string = string.replace("``` c", "```")
        string = string.replace("```c", "```")
        string = string.replace("```C", "```")
        string = string.split("```")[1].split("```")[0]
    string = string.replace("threadIdx.x", "threadIdx_x")
    string = string.replace("blockIdx.x", "blockIdx_x")
    string = string.replace("__mlu_global__ ", "")
    string = string.replace("__global__ ", "")
    return string


def auto_debug_C(target_file):
    max_attempts = 10
    for attempt in range(1, max_attempts + 1):
        success, output = run_C_test_file(target_file)

        if hasattr(output, "stdout") and "验证通过" in output.stdout:
            print("The C unittest ran successfully!")
            break
        else:
            with open(target_file, "r") as f:
                error_code = f.read()
                f.close()
            if "OSError" in output:
                print("[INFO]**************error_msg:", output)
                print(
                    f"Attempt {attempt}: Error encountered while running the script"
                )

                try:
                    fixed_code = fix_compilation_code(error_code, output)
                    fixed_code = extract_C_code(fixed_code)
                    print(f"gpt suggested fix:\n {fixed_code}")

                    with open(target_file, "w") as f:
                        f.write(fixed_code)
                        f.close()
                except BaseException:
                    continue

            elif "AssertionError" in output:
                print("The C unittest failed!")
                # find the path
                try:
                    fixed_code = fix_computation_code(
                        error_code, output.split("output"[1])
                    )
                    fixed_code = extract_C_code(fixed_code)
                    print(f"gpt suggested fix:\n {fixed_code}")
                    with open(target_file, "w") as f:
                        f.write(fixed_code)
                        f.close()
                except BaseException:
                    continue
    if not success:
        print(
            "Maximum number of attempts reached. Please try fixing the script manually or run AutoDebug again."
        )


if __name__ == "__main__":
    test_files = glob.glob("zeroshot_data_gen/bang_2_cpp/*.cpp")
    for test_file in test_files:
        with open(test_file, "r") as f:
            error_code = f.read()
            f.close()

        error_code = (
            """extern "C" __mlu_global__ void""" + error_code.split("void")[1]
        )
        base_name = os.path.basename(test_file)
        target_file = os.path.join(
            "selfdebug_data_gen/zero_shot/bang_2_cpp", base_name
        )
        if os.path.exists(target_file):
            continue

        with open(target_file, "w") as f:
            f.write(error_code)
            f.close()

        auto_debug_C(target_file)
