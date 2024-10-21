import re
import openai

from src.prompt.prompt import SYSTEM_PROMPT
from src.pre_processing.preprocessing_prompt import (
    LOOP_RECOVERY_PROMPT_CUDA,
    LOOP_RECOVERY_DEMO_CUDA,
    LOOP_RECOVERY_PROMPT_BANG,
    LOOP_RECOVERY_DEMO_BANG,
    DETENSORIZATION_PROMPT_BANG,
)

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
    match = re.search(r"\`\`\`(.*?)\`\`\`", content, re.DOTALL)
    if match:
        code_content = match.group(1)
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
    match = re.search(r"\`\`\`(.*?)\`\`\`", content, re.DOTALL)
    if match:
        code_content = match.group(1)
        print(code_content)
        return code_content
    return None


def extract_bang_instructions(code):
    # Define a regex pattern to match instructions starting with __bang
    pattern = r"__bang\w+"
    # Find all matches in the provided code
    instructions = re.findall(pattern, code)
    return instructions


op_dict = {
    "__memcpy": """void __memcpy(void *dst, const void *src, unsigned int size, mluMemcpyDirection_t dir)
    Copies <size> bytes data from source address <src> to destination address <dst>.

    parameters:
        [out] dst: The address of destination area.

        [in] src: The address of source area.

        [in] size: The number of bytes to be copied.

        [in] dir: The copy direction.
    """,
    "__bang_active_tanh": """void __bang_active_tanh(float *dst, const float *src, unsigned int elem_count)
    Applies active (tanh) operation on <src>.

    The function requires auxiliary __nram__ space internally. See the table Activation Table Space for the Activation Function for more information.

    Parameters
        [out] dst: The address of the destination vector.

        [in] src: The address of the source vector.

        [in] elem_count: Number of elements in the source vector.
    """,
}


def run_detensorization(code, target):
    instructions = extract_bang_instructions(code)
    # First, detensorize memory
    code = detensorization("__memcpy", code, op_dict["__memcpy"])
    for inst in instructions:
        code = detensorization(inst, code, op_dict[inst])
    return code


def pre_processing_pipeline(code, target):
    """This function transforms the given code by performing two main transformations:
        1. Convert parallel loop variables (e.g., OpenMP, CUDA) into standard C for loops.
        2. Convert SIMD tensor operations into scalar for-loop based calculations.
    :param func_content: The content of the function (code) to be transformed.
    :return: Transformed code after applying the two transformations."""
    code = run_loop_recovery(code, target)
    code = run_detensorization(code, target)
    return code
