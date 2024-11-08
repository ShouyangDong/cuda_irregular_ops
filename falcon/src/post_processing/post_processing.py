import json
import re

import openai

from falcon.src.post_processing.post_processing_prompt import (
    CACHE_READ_DEMO,
    CACHE_READ_PROMPT,
    CACHE_WRITE_DEMO,
    CACHE_WRITE_PROMPT,
    DECORATION_PROMPT,
    TENSORIZATION_PROMPT,
    THREAD_BINDING_DEMO_BANG,
    THREAD_BINDING_DEMO_CUDA,
    THREAD_BINDING_PROMPT_BANG,
    THREAD_BINDING_PROMPT_CUDA,
)
from falcon.src.prompt.prompt import SYSTEM_PROMPT

model_name = """gpt-4-turbo"""
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
    vairables = None
    THREAD_BINDING_PROMPT = None
    if target == "CUDA":
        prompt_demo = THREAD_BINDING_DEMO_CUDA
        THREAD_BINDING_PROMPT = THREAD_BINDING_PROMPT_CUDA
    elif target == "BANG":
        prompt_demo = THREAD_BINDING_DEMO_BANG
        THREAD_BINDING_PROMPT = THREAD_BINDING_PROMPT_BANG

    PROMPT = PROMPT.replace("{THREAD_BINDING_PROMPT}", THREAD_BINDING_PROMPT)
    PROMPT = PROMPT.replace("{THREAD_BINDING_DEMO}", prompt_demo)
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


def get_operation_content(code):
    # Define the regex pattern to match the content inside parentheses following #pragma intrinsic
    pattern = r"#pragma\s+operation\(([^)]+)\)"
    # Find all matches in the given code (there might be multiple pragmas)
    matches = re.findall(pattern, code)
    return matches


def get_input_operand(pragma):
    inputs = pragma.split("input[")[1].split("]")[0]
    input_list = inputs.split(", ")
    return input_list


def get_output_operand(pragma):
    outputs = pragma.split("output[")[1].split("]")[0]
    output_list = outputs.split(", ")
    return output_list


def replace_operation_with_intrinsic(code, op_pragma):
    # Get the list of operations from the code
    op_list = get_operation_content(code)
    space_maps = []
    # Iterate over each operation found in the code
    for op in op_list:
        op_name = op.split("(input")[0]
        # Build the pragma search pattern to match the specific operation pragma
        pragma_pattern = re.escape(f"#pragma operation({op})")
        # Ensure the operation exists in the op_pragma dictionary
        if op_name not in op_pragma:
            raise KeyError(f"Operation '{op_name}' not found in op_pragma.")
        # Get input and output operands for the operation
        input_operands = get_input_operand(op)
        output_operands = get_output_operand(op)
        # Get corresponding memory spaces from the op_pragma dictionary
        input_spaces = get_input_operand(op_pragma[op_name])
        output_spaces = get_output_operand(op_pragma[op_name])
        # Ensure the input/output operand lists and spaces have matching lengths
        if len(input_operands) != len(input_spaces):
            raise ValueError(
                f"Input operands and memory spaces length mismatch for operation '{op_name}' "
                f"({len(input_operands)} operands vs {len(input_spaces)} spaces)."
            )

        if len(output_operands) != len(output_spaces):
            raise ValueError(
                f"Output operands and memory spaces length mismatch for operation '{op_name}' "
                f"({len(output_operands)} operands vs {len(output_spaces)} spaces)."
            )
        # Create the space map by zipping operands and spaces
        input_map = {
            operand: space for operand, space in zip(input_operands, input_spaces)
        }
        output_map = {
            operand: space for operand, space in zip(output_operands, output_spaces)
        }
        space_map = {"input": input_map, "output": output_map}
        # Replace the matching pragma with the corresponding value from op_pragma
        code = re.sub(pragma_pattern, op_pragma[op_name], code)
        # Append the space map for this operation
        space_maps.append(space_map)
    return code, space_maps


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
    pattern = r"#pragma\s+operation\((\w+)"
    # Find all matches in the given string
    matches = re.findall(pattern, pragma_line)
    # Return the list of matches (it will be empty if no matches are found)
    return matches


def run_tensorization(code, target):
    op_dict = json.load(open("./falcon/documents/bang_c_op_map.json", "r"))
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
        return code_content.replace("cpp", "")
    return None


def post_processing_pipeline(code, target):
    """This function transforms the given code by performing two main transformations:
        1. Convert parallel loop variables (e.g., OpenMP, CUDA) into standard C for loops.
        2. Convert SIMD tensor operations into scalar for-loop based calculations.
    :param func_content: The content of the function (code) to be transformed.

    :return: Transformed code after applying the two transformations."""
    code = run_thread_binding(code, target)

    # when target is "BANG" or "DLBOOST", insert tensorization process.
    if target in ["BANG", "DLBOOST"]:
        code = run_code_decoration(code)
        op_pragma = {}
        if target == "BANG":
            op_pragma = json.load(
                open("./falcon/documents/operation_bang_C_instruction_map.json", "r")
            )
        code, space_maps = replace_operation_with_intrinsic(code, op_pragma)
        code = run_cache_process(code, space_maps)
        code = run_code_decoration(code)
        code = run_tensorization(code, target)
    return code
