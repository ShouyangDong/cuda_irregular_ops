import os
import re

import openai

model_name = """gpt-4-turbo"""
openai.api_key = "sk-JmlwEmWiNtFqSD7IDaF981Dd8a7447FfBcE768755cB38010"
openai.api_base = "https://api.keya.pw/v1"


def run_transcompile(code):
    PROMPT = """Please rewrite the following DLBoost-optimized code into CUDA C code for GPU acceleration. Ensure the converted code retains the same functionality and is optimized for CUDA's parallel processing capabilities.

    Requirements:

    1. Replace DLBoost-specific logic with CUDA kernels and GPU-compatible operations;
    2. Optimize the code for efficient memory usage and parallel execution on CUDA-enabled GPUs;
    3. Provide comments explaining how the CUDA implementation corresponds to the original DLBoost functionality;
    4. Maintain the input-output interfaces and preserve the original computational intent.
    Original DLBoost Code:
    {code}
    """
    PROMPT = PROMPT.replace("{code}", code)
    transformation_completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "user", "content": PROMPT}],
    )

    content = transformation_completion.choices[0].message["content"]
    match = re.search(r"```[a-zA-Z]*\n(.*?)```", content, re.S)
    if match:
        code_content = match.group(1).strip()
        return code_content
    return None


if __name__ == "__main__":
    files = glob.glob("benchmark/data/dlboost_code_test/*.cpp")
    for file in tqdm(files):
        base_name = os.path.basename(file)
        with open(file, "r") as f:
            source = f.read()
            f.close()

        target_code = run_transcompile(source)
        file_name = os.path.join(
            "benchmark/transcompilation/dlboost/cuda", base_name
        )
        with open(file_name, mode="w") as f:
            f.write(target_code)
            f.close()
