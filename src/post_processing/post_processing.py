import re
import openai

from src.prompt.prompt import SYSTEM_PROMPT
from src.post_processing.post_processing_prompt import (
    THREAD_BINDING_DEMO_BANG,
    THREAD_BINDING_DEMO_CUDA,
    THREAD_BINDING_PROMPT,
    CACHE_READ_PROMPT,
    CACHE_READ_DEMO,
    CACHE_WRITE_PROMPT,
    CACHE_WRITE_DEMO,
    DECORATION_PROMPT,
    TENSORIZATION_PROMPT,
)

model_name = """gpt-3.5-turbo"""
openai.api_key = "sk-JmlwEmWiNtFqSD7IDaF981Dd8a7447FfBcE768755cB38010"
openai.api_base = "https://api.keya.pw/v1"


def run_thread_binding(code, target):
    PROMPT = """
    {SYSTEM_PROMPT}
    
    {THREAD_BINDING_PROMPT}
    
    Please return the output kernel function without any additional information.
    """

    PROMPT = PROMPT.replace("{SYSTEM_PROMPT}", SYSTEM_PROMPT)
    prompt_demo = None
    if target == "CUDA":
        prompt_demo = THREAD_BINDING_DEMO_CUDA
    elif target == "BANG":
        prompt_demo = THREAD_BINDING_DEMO_BANG

    PROMPT = PROMPT.replace("{THREAD_BINDING_PROMPT}", THREAD_BINDING_PROMPT)
    PROMPT = PROMPT.replace("{LOOP_RECOVERY_DEMO}", prompt_demo)
    PROMPT = PROMPT.replace("{cpp_code}", code)
    transformation_completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "user", "content": PROMPT}],
    )

    content = transformation_completion.choices[0].message["content"]
    match = re.search(r"\`\`\`(.*?)\`\`\`", content, re.DOTALL)
    if match:
        code_content = match.group(1)
        return code_content
    return None


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


def run_cache_process(code):
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


def tensorization(op, code, document):
    PROMPT = """
    {SYSTEM_PROMPT}
    
    Here is the introduction of Tensorization: {TENSORIZATION_PROMPT}
    Please tensorize the sequential code of {op} below the #pragma operation in {code} 
    accordingt to the introduction of tensorized intrinsic.
    {document}
    Please return the output kernel function without any additional information.
    """

    PROMPT = PROMPT.replace("{SYSTEM_PROMPT}", SYSTEM_PROMPT)
    PROMPT = PROMPT.replace("{TENSORIZATION_PROMPT}", TENSORIZATION_PROMPT)
    PROMPT = PROMPT.replace("{document}", document)
    PROMPT = PROMPT.replace("{code}", code)
    PROMPT = PROMPT.replace("{op}", op)

    transformation_completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "user", "content": PROMPT}],
    )

    content = transformation_completion.choices[0].message["content"]
    match = re.search(r"\`\`\`(.*?)\`\`\`", content, re.DOTALL)
    if match:
        code_content = match.group(1)
        return code_content
    return None


def get_operation_words(pragma_line):
    # Modify the pattern to capture everything inside parentheses, allowing spaces and underscores
    pattern = r"#pragma\s+operation\(([^)]+)\)"
    # Find all matches in the given string
    matches = re.findall(pattern, pragma_line)
    # Return the list of matches (it will be empty if no matches are found)
    return matches


op_dict = {
    "memcpy": """void __memcpy(void *dst, const void *src, unsigned int size, mluMemcpyDirection_t dir)
    Copies <size> bytes data from source address <src> to destination address <dst>.

    parameters:
        [out] dst: The address of destination area.

        [in] src: The address of source area.

        [in] size: The number of bytes to be copied.

        [in] dir: The copy direction.
    """,
    "matmul": """void __bang_mlp(float *dst, const float *src, const float *filter, int height, int width)
    Applies multilayer perception operation. <dst>=<src>×<filter>.

    Parameters
    [out] dst: The address of the destination vector.

    [in] src: The address of the source vector.

    [in] filter: The address of filter matrix which has row-major data layout.

    [in] height: The height of <filter>.

    [in] width: The width of <filter>.
    """,
    "add": """void __bang_add(float *dst, const float *src0, const float *src1, unsigned int elem_count)
    This function performs addition operation element-wisely on <src0> and <src1> and saves the result in <dst>.

    Parameters
    [out] dst: The address of the destination vector.

    [in] src0: The address of the first source vector.

    [in] src1: The address of the second source vector.

    [in] elem_count: The number of elements in the source vector.
    """,
}


def run_tensorization(code, target):
    op_list = get_operation_words(code)
    for op in op_list:
        op = "memcpy" if "memory" in op else op
        op_document = op_dict[op]
        code = tensorization(op, code, op_document)
    return code


def run_code_decoration(code):
    PROMPT = DECORATION_PROMPT.replace("{cpp_code}", code)
    decoration_completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "user", "content": PROMPT}],
    )

    content = decoration_completion.choices[0].message["content"]

    match = re.search(r"\`\`\`(.*?)\`\`\`", content, re.DOTALL)
    if match:
        code_content = match.group(1)
        return code_content
    return None


def post_processing_pipeline(code, target):
    """This function transforms the given code by performing two main transformations:
        1. Convert parallel loop variables (e.g., OpenMP, CUDA) into standard C for loops.
        2. Convert SIMD tensor operations into scalar for-loop based calculations.
    :param func_content: The content of the function (code) to be transformed.

    :return: Transformed code after applying the two transformations."""
    code = run_thread_binding(code, target)
    code = run_code_decoration(code)
    code = run_cache_process(code)
    code = run_code_decoration(code)
    code = run_tensorization(code, target)
    return code
