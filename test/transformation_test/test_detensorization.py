import re
import openai

from src.prompt.prompt import SYSTEM_PROMPT
from src.pre_processing.preprocessing_prompt import (
    DETENSORIZATION_PROMPT_BANG,
)

model_name = """gpt-3.5-turbo"""
openai.api_key = "sk-JmlwEmWiNtFqSD7IDaF981Dd8a7447FfBcE768755cB38010"
openai.api_base = "https://api.keya.pw/v1"


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


if __name__ == "__main__":
    code = """
    extern "C" __mlu_global__ void tanh(float* input0, float* active_tanh_210) {
        __nram__ float input0_local_nram[640];
        __memcpy(((float *)input0_local_nram + (0)), ((float *)input0 + (((((int)clusterId) * 2560) + (((int)coreId) * 640)))), 2560, GDRAM2NRAM);
        __bang_active_tanh(((float *)input0_local_nram + (0)), ((float *)input0_local_nram + (0)), 640);
        __memcpy(((float *)active_tanh_210 + (((((int)clusterId) * 2560) + (((int)coreId) * 640)))), ((float *)input0_local_nram + (0)), 2560, NRAM2GDRAM);
    }
    """
    code = run_detensorization(code, target="BANG")
    print(code)
