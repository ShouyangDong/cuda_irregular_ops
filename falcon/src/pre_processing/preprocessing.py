import json
import re

import openai

from falcon.buffer_inline import ast_buffer_inline
from falcon.simplification import simplify_code
from falcon.src.pre_processing.preprocessing_prompt import (
    DETENSORIZATION_PROMPT_BANG,
    LOOP_RECOVERY_DEMO_BANG,
    LOOP_RECOVERY_DEMO_CUDA,
    LOOP_RECOVERY_PROMPT_BANG,
    LOOP_RECOVERY_PROMPT_CUDA,
)
from falcon.src.prompt.prompt import SYSTEM_PROMPT
from falcon.stmt_simplification import ast_stmt_simplification
from falcon.util import make_full_func

model_name = """gpt-4-turbo"""
openai.api_key = "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"


def run_loop_recovery(code, target):
    PROMPT = """
    {SYSTEM_PROMPT}

    {TENSORIZATION_PROMPT}

    Example:
    {LOOP_RECOVERY_DEMO}

    Input cuda Code:
    {code}
    Output C++ Code:

    Please return the output kernel function without any additional information.
    """

    PROMPT = PROMPT.replace("{SYSTEM_PROMPT}", SYSTEM_PROMPT)
    prompt_des = None
    if target == "cuda" or target == "hip":
        prompt_des = LOOP_RECOVERY_PROMPT_CUDA
    elif target == "mlu":
        prompt_des = LOOP_RECOVERY_PROMPT_BANG
    prompt_demo = None
    if target == "cuda" or target == "hip":
        prompt_demo = LOOP_RECOVERY_DEMO_CUDA
    elif target == "mlu":
        prompt_demo = LOOP_RECOVERY_DEMO_BANG

    PROMPT = PROMPT.replace("{TENSORIZATION_PROMPT}", prompt_des)
    PROMPT = PROMPT.replace("{LOOP_RECOVERY_DEMO}", prompt_demo)
    PROMPT = PROMPT.replace("{code}", code)
    transformation_completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "user", "content": PROMPT}],
    )

    content = transformation_completion.choices[0].message["content"]
    match = re.search(r"```[a-zA-Z]*\n(.*?)```", content, re.S)
    if match:
        code_content = match.group(1).strip()
        code_content = code_content.replace("coreId", "core_id")
        code_content = code_content.replace("clusterId", "cluster_id")
        code_content = make_full_func(code_content, target)
        return code_content
    return None


def detensorization(op, code, document):
    PROMPT = """
    {SYSTEM_PROMPT}

    Here is the introduction of Detensorization: {DETENSORIZATION_PROMPT_BANG}
    Please transform the instruction {op} in following code into sequential for loop.

    {code}

    accordingt to the description of tinstruction.

    {document}

    Please return the output kernel function without any additional information.
    """

    PROMPT = PROMPT.replace("{SYSTEM_PROMPT}", SYSTEM_PROMPT)
    PROMPT = PROMPT.replace(
        "{DETENSORIZATION_PROMPT_BANG}", DETENSORIZATION_PROMPT_BANG
    )
    PROMPT = PROMPT.replace("{document}", document)
    PROMPT = PROMPT.replace("{code}", code)
    PROMPT = PROMPT.replace("{op}", op)
    # print(PROMPT)
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


def extract_bang_instructions(code):
    # Define a regex pattern to match instructions starting with __bang
    pattern = r"__bang\w+"
    # Find all matches in the provided code
    instructions = re.findall(pattern, code)
    return instructions


def run_detensorization(code, target):
    if target == "mlu":
        op_dict = json.load(
            open("./falcon/documents/bang_c_user_guide.json", "r")
        )
        instructions = extract_bang_instructions(code)
        if "__memcpy" in code:
            code = detensorization("__memcpy", code, op_dict["__memcpy"])

        if instructions is not None:
            for inst in instructions:
                code = detensorization(inst, code, op_dict[inst])

    code = simplify_code(code)
    code = ast_stmt_simplification(code)
    code = ast_buffer_inline(code)
    code = make_full_func(code, target)
    return code


def pre_processing_pipeline(code, target):
    """This function transforms the given code by performing two main transformations:
        1. Convert parallel loop variables (e.g., OpenMP, cuda) into standard C for loops.
        2. Convert SIMD tensor operations into scalar for-loop based calculations.
    :param func_content: The content of the function (code) to be transformed.
    :return: Transformed code after applying the two transformations."""
    code = run_loop_recovery(code, target)
    if target in ["mlu"]:
        code = run_detensorization(code, target)
    return code
