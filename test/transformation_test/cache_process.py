import re
import openai

from src.post_processing.post_processing_prompt import (
    CACHE_READ_PROMPT,
    CACHE_READ_DEMO,
    CACHE_WRITE_PROMPT,
    CACHE_WRITE_DEMO,
)
from src.prompt.prompt import SYSTEM_PROMPT


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


def generate_cache_read_prompt(i, space, op_name, code):
    PROMPT = """
    {SYSTEM_PROMPT}
    
    Here is the introduction of cache read: {CACHE_READ_PROMPT}
    The {i}st input argument is readed into {space} memory space.
    Please transform the following code {code} accordingt to the demo:
    {CACHE_READ_DEMO}"""

    PROMPT = PROMPT.replace("{SYSTEM_PROMPT}", SYSTEM_PROMPT)
    PROMPT = PROMPT.replace("{CACHE_READ_PROMPT}", CACHE_READ_PROMPT)
    PROMPT = PROMPT.replace("{i}", str(i))
    PROMPT = PROMPT.replace("{space}", space)
    PROMPT = PROMPT.replace("{CACHE_READ_DEMO}", CACHE_READ_DEMO)
    PROMPT = PROMPT.replace("{code}", code)
    return PROMPT


def generate_cache_write_prompt(i, space, op_name, code):
    PROMPT = """
    {SYSTEM_PROMPT}

    Here is the introduction of cache write: {CACHE_WRITE_PROMPT}
    The {i}st output argument is writed into {space} memory space.
    Please transform the following code {code} accordingt to the demo:
    {CACHE_WRITE_DEMO}"""

    PROMPT = PROMPT.replace("{SYSTEM_PROMPT}", SYSTEM_PROMPT)
    PROMPT = PROMPT.replace("{CACHE_WRITE_PROMPT}", CACHE_WRITE_PROMPT)
    PROMPT = PROMPT.replace("{i}", str(i))
    PROMPT = PROMPT.replace("{space}", space)
    PROMPT = PROMPT.replace("{CACHE_WRITE_DEMO}", CACHE_WRITE_DEMO)
    PROMPT = PROMPT.replace("{code}", code)
    return PROMPT


def do_cache_process(code):
    # Get the list of intrinsics from the code
    intrinsic_list = get_intrinsic_content(code)
    # Iterate over each intrinsic found in the code
    for op in intrinsic_list:
        op_name = op.split("__")[1].split("(")[0]
        inputs = get_input_memory_spaces(op)
        outputs = get_output_memory_spaces(op)
        for i, space in enumerate(inputs):
            cache_read_prompt = generate_cache_read_prompt(i + 1, space, op_name, code)
            print("[INFO]*************cache_read_prompt: ", cache_read_prompt)

        for i, space in enumerate(outputs):
            cache_write_prompt = generate_cache_write_prompt(
                i + 1, space, op_name, code
            )
            print("[INFO]*************cache_write_prompt: ", cache_write_prompt)
    return code


if __name__ == "__main__":
    code = """
    #pragma intrinsic(__bang_mlp(input[Nram, Wram], output[Nram]))
    for (int col = 0; col < 64; col++) {
        C[(clusterId * 4 + coreId) * 64 + col] = 0.0f;
        for (int i = 0; i < 512; i++) {
            C[col] += A[i] * B[i * 64 + col];
        }
    }
    
    #pragma intrinsic(__bang_add(input[Nram], output[Nram]))
    for (int i = 0; i < 512; i++) {
        C[i] = A[i] + B[i];
    }

    """
    final_code = do_cache_process(code)
