import re
import openai
import json
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
    The {i}st input argument is readed into {CACHE_NAME} memory space.

    Please transform the following code {code} accordingt to the Example:
    {CACHE_READ_DEMO}
    Please return the output kernel function without any additional information.
    """
    space_map = {"nram": "__nram__", "wram": "__wram__"}
    NAMESPACE = space_map[space.lower()]

    PROMPT = PROMPT.replace("{SYSTEM_PROMPT}", SYSTEM_PROMPT)
    PROMPT = PROMPT.replace("{CACHE_READ_PROMPT}", CACHE_READ_PROMPT)
    PROMPT = PROMPT.replace("{i}", str(i))
    PROMPT = PROMPT.replace("{CACHE_READ_DEMO}", CACHE_READ_DEMO)
    PROMPT = PROMPT.replace("{CACHE_NAME}", space)
    PROMPT = PROMPT.replace("{code}", code)
    PROMPT = PROMPT.replace("{NAMESPACE}", NAMESPACE)
    return PROMPT


def generate_cache_write_prompt(i, space, op_name, code):
    assert space, "memory space cannot be empty"
    PROMPT = """
    {SYSTEM_PROMPT}

    Here is the introduction of cache write: {CACHE_WRITE_PROMPT}
    The {i}st output argument is writed into {CACHE_NAME} memory space.
    Please transform the following code {code} accordingt to the demo:
    {CACHE_WRITE_DEMO}
    Please return the output kernel function without any additional information.
    """
    NAMESPACE = "__nram__"
    PROMPT = PROMPT.replace("{SYSTEM_PROMPT}", SYSTEM_PROMPT)
    PROMPT = PROMPT.replace("{CACHE_WRITE_PROMPT}", CACHE_WRITE_PROMPT)
    PROMPT = PROMPT.replace("{i}", str(i))
    PROMPT = PROMPT.replace("{CACHE_WRITE_DEMO}", CACHE_WRITE_DEMO)
    PROMPT = PROMPT.replace("{CACHE_NAME}", space)
    PROMPT = PROMPT.replace("{code}", code)
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
            match = re.search(r"\`\`\`(.*?)\`\`\`", content, re.DOTALL)
            code = match.group(1) if match else code
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


def run_tensorization(code, target):
    op_dict = json.load(open("./documents/bang_c_user_guide", "r"))
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
    op_pragma = {}
    final_code, space_maps = replace_operation_with_intrinsic(code, op_pragma)
    code = run_cache_process(code, space_maps)
    code = run_code_decoration(code)
    code = run_tensorization(code, target)
    return code
