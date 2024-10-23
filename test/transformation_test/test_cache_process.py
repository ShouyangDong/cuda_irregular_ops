import re
import openai

from src.post_processing.post_processing_prompt import (
    CACHE_READ_PROMPT,
    CACHE_READ_DEMO,
    CACHE_WRITE_PROMPT,
    CACHE_WRITE_DEMO,
)
from src.prompt.prompt import SYSTEM_PROMPT

model_name = """gpt-4-turbo"""
openai.api_key = "sk-JmlwEmWiNtFqSD7IDaF981Dd8a7447FfBcE768755cB38010"
openai.api_base = "https://api.keya.pw/v1"


def get_intrinsic_content(code):
    # Define the regex pattern to match the content inside parentheses following #pragma intrinsic
    pattern = r"#pragma\s+intrinsic\(([^)]+)\)"
    # Find all matches in the given code (there might be multiple pragmas)
    matches = re.findall(pattern, code)
    return matches


def get_input_memory_spaces(pragma):
    inputs = pragma.split("input[")[1].split("]")[0]
    input_list = inputs.split(", ")
    return input_list


def get_output_memory_spaces(pragma):
    outputs = pragma.split("output[")[1].split("]")[0]
    output_list = outputs.split(", ")
    return output_list


def generate_cache_read_prompt(buffer, space, code):
    PROMPT = """
    {SYSTEM_PROMPT}
    
    {CACHE_READ_PROMPT}

    {CACHE_READ_DEMO}
    Please return the output kernel function without any additional information.
    """
    space_map = {"nram": "__nram__", "wram": "__wram__"}
    NAMESPACE = space_map[space.lower()]

    PROMPT = PROMPT.replace("{SYSTEM_PROMPT}", SYSTEM_PROMPT)
    PROMPT = PROMPT.replace("{CACHE_READ_PROMPT}", CACHE_READ_PROMPT)
    PROMPT = PROMPT.replace("{buffer}", buffer)
    PROMPT = PROMPT.replace("{CACHE_READ_DEMO}", CACHE_READ_DEMO)
    PROMPT = PROMPT.replace("{CACHE_NAME}", space)
    PROMPT = PROMPT.replace("{CODE}", code)
    PROMPT = PROMPT.replace("{NAMESPACE}", NAMESPACE)
    return PROMPT


def generate_cache_write_prompt(buffer, space, code):
    assert space, "memory space cannot be empty"
    PROMPT = """
    {SYSTEM_PROMPT}
    {CACHE_WRITE_PROMPT}
    {CACHE_WRITE_DEMO}
    Please return the output kernel function without any additional information.
    """
    NAMESPACE = "__nram__"
    PROMPT = PROMPT.replace("{SYSTEM_PROMPT}", SYSTEM_PROMPT)
    PROMPT = PROMPT.replace("{CACHE_WRITE_PROMPT}", CACHE_WRITE_PROMPT)
    PROMPT = PROMPT.replace("{buffer}", buffer)
    PROMPT = PROMPT.replace("{CACHE_WRITE_DEMO}", CACHE_WRITE_DEMO)
    PROMPT = PROMPT.replace("{CACHE_NAME}", space)
    PROMPT = PROMPT.replace("{CODE}", code)
    PROMPT = PROMPT.replace("{NAMESPACE}", NAMESPACE)
    return PROMPT


def run_cache_process(code, space_maps):
    # Get the list of intrinsics from the code
    intrinsic_list = get_intrinsic_content(code)
    # Ensure the intrinsic lists and spaces have matching lengths
    if len(intrinsic_list) != len(space_maps):
        raise ValueError(
            f"intrinsics and memory spaces length mismatch for operation"
            f"({len(intrinsic_list)} intrinsics vs {len(space_maps)} spaces)."
        )
    # Iterate over each intrinsic found in the code
    for op, space_map in zip(intrinsic_list, space_maps):
        for key, value in space_map["input"].items():
            cache_read_prompt = generate_cache_read_prompt(key, value, code)
            transformation_completion = openai.ChatCompletion.create(
                model=model_name,
                messages=[{"role": "user", "content": cache_read_prompt}],
            )
            content = transformation_completion.choices[0].message["content"]
            match = re.search(r"\`\`\`(.*?)\`\`\`", content, re.DOTALL)
            code = match.group(1) if match else code

        for key, value in space_map["output"].items():
            cache_write_prompt = generate_cache_write_prompt(key, value, code)
            transformation_completion = openai.ChatCompletion.create(
                model=model_name,
                messages=[{"role": "user", "content": cache_write_prompt}],
            )
            content = transformation_completion.choices[0].message["content"]
            print("[IFNO]**********content: ", content)
            match = re.search(r"\`\`\`(.*?)\`\`\`", content, re.DOTALL)
            code = match.group(1) if match else code
    return code


if __name__ == "__main__":
    code = """
    #pragma intrinsic(__bang_add(input[Nram, Nram], output[Nram]))
    for (int i = 0; i < 512; i++) {
        C[i] = A[i] + B[i];
    }

    """
    space_maps = [
        {"input": {"A": "Nram", "B": "Nram"}, "output": {"C": "Nram"}},
    ]

    final_code = run_cache_process(code, space_maps)
    print(final_code)
