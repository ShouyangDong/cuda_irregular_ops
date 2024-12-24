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

model_name = """gpt-4-turbo"""
openai.api_key = "sk-JmlwEmWiNtFqSD7IDaF981Dd8a7447FfBcE768755cB38010"
openai.api_base = "https://api.keya.pw/v1"


def run_loop_recovery(code, target):
    PROMPT = """
    {SYSTEM_PROMPT}

    {TENSORIZATION_PROMPT}

    Example:
    {LOOP_RECOVERY_DEMO}

    Input CUDA Code:
    {code}
    Output C++ Code:

    Please return the output kernel function without any additional information.
    """

    PROMPT = PROMPT.replace("{SYSTEM_PROMPT}", SYSTEM_PROMPT)
    prompt_des = None
    if target == "CUDA":
        prompt_des = LOOP_RECOVERY_PROMPT_CUDA
    elif target == "BANG":
        prompt_des = LOOP_RECOVERY_PROMPT_BANG
    prompt_demo = None
    if target == "CUDA":
        prompt_demo = LOOP_RECOVERY_DEMO_CUDA
    elif target == "BANG":
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
        return code_content
    return None


def detensorization(op, code, document):
    print("[INFO]***********code:", code)
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
    op_dict = json.load(open("./falcon/documents/bang_c_user_guide.json", "r"))
    instructions = extract_bang_instructions(code)
    if "__memcpy" in code:
        # First, detensorize memory
        code = detensorization("__memcpy", code, op_dict["__memcpy"])

    if instructions is not None:
        for inst in instructions:
            code = detensorization(inst, code, op_dict[inst])

    code = simplify_code(code)
    code = ast_stmt_simplification(code)
    code = ast_buffer_inline(code)
    return code


def pre_processing_pipeline(code, target):
    """This function transforms the given code by performing two main transformations:
        1. Convert parallel loop variables (e.g., OpenMP, CUDA) into standard C for loops.
        2. Convert SIMD tensor operations into scalar for-loop based calculations.
    :param func_content: The content of the function (code) to be transformed.
    :return: Transformed code after applying the two transformations."""
    code = run_loop_recovery(code, target)
    if target in ["BANG"]:
        code = run_detensorization(code, target)
    return code
