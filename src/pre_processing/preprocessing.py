import re
import openai

from src.pre_processing.preprocessing_prompt import *
from src.prompt.prompt import SYSTEM_PROMPT, PRAGMA_INSERT_PROMPT, APPLY_OPT_PROMPT

OPT_LIST = ["LOOP_RECOVERY", "DETENSORIZATION"]
model_name = """gpt-3.5-turbo"""
openai.api_key = "sk-JmlwEmWiNtFqSD7IDaF981Dd8a7447FfBcE768755cB38010"
openai.api_base = "https://api.keya.pw/v1"

PRE_PROCESSING_PRAGMA_HINTS = {
    "LOOP_RECOVERY": "#pragma thread({target})",
    "DETENSORIZATION": "#pragma intrinsic({target})",
}


def run_code_analysis(code, pass_name, target):
    PRAGMA_DESCRIPTION = globals()[pass_name + "_PROMPT_" + target]
    _PRAGMA_INSERT_PROMPT = PRAGMA_INSERT_PROMPT.replace(
        "{PRAGMA_DESCRIPTION}", PRAGMA_DESCRIPTION
    )
    _PRAGMA_INSERT_PROMPT = _PRAGMA_INSERT_PROMPT.replace(
        "{PRAGMA_NAME}",
        PRE_PROCESSING_PRAGMA_HINTS[pass_name].replace("{target}", target),
    )
    _PRAGMA_INSERT_PROMPT = _PRAGMA_INSERT_PROMPT.replace("{STAGE_CODE_CONTENT}", code)

    STAGE_OPT_PROMPT_COMPLETE = SYSTEM_PROMPT + _PRAGMA_INSERT_PROMPT
    analysis_completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "user", "content": STAGE_OPT_PROMPT_COMPLETE}],
    )
    content = analysis_completion.choices[0].message["content"]
    print("[INFO]*********final code: ", content)
    match = re.search(r"\`\`\`(.*?)\`\`\`", content, re.DOTALL)

    return match


def run_code_transformation(code, pass_name, pragma):
    PRAGMA_DEMO_COMPLETE = globals()[pass_name]
    _APPLY_OPT_PROMPT = APPLY_OPT_PROMPT.replace("{STAGE_CODE_CONTENT}", func_content)
    _APPLY_OPT_PROMPT = _APPLY_OPT_PROMPT.replace("{OPT_LIST}", trans)
    _APPLY_OPT_PROMPT = _APPLY_OPT_PROMPT.replace("{PRAGMA_DEMO}", PRAGMA_DEMO_COMPLETE)

    STAGE_OPT_PROMPT_COMPLETE = SYSTEM_PROMPT + _APPLY_OPT_PROMPT
    transformation_completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "user", "content": STAGE_OPT_PROMPT_COMPLETE}],
    )
    content = chat_completion.choices[0].message["content"]
    match = re.search(r"\`\`\`(.*?)\`\`\`", content, re.DOTALL)
    return match


def pre_processing_pipeline(func_content, target):
    """This function transforms the given code by performing two main transformations:
        1. Convert parallel loop variables (e.g., OpenMP, CUDA) into standard C for loops.
        2. Convert SIMD tensor operations into scalar for-loop based calculations.
    :param func_content: The content of the function (code) to be transformed.
    :return: Transformed code after applying the two transformations."""
    TRANS_DESCRIPTION = ""
    for i, trans in enumerate(OPT_LIST):
        prompt_name = (
            f"{trans}_PROMPT_{target}"
            if trans != "DETENSORIZATION"
            else f"{trans}_PROMPT"
        )

        # First analysis the code, and insert corresponding pragma
        pragma_code = run_code_analysis(func_content, trans, target)
        # # Transform the code according to the pragma
        # transform_code = run_code_transformation(pragma_code, trans, target)
    return pragma_code


if __name__ == "__main__":
    func_content = """
    extern "C" __mlu_global__ void tanh(float* input0, float* active_tanh_210) {
        __nram__ float input0_local_nram[640];
        __memcpy(((float *)input0_local_nram + (0)), ((float *)input0 + (((((int)clusterId) * 2560) + (((int)coreId) * 640)))), 2560, GDRAM2NRAM);
        __bang_active_tanh(((float *)input0_local_nram + (0)), ((float *)input0_local_nram + (0)), 640);
        __memcpy(((float *)active_tanh_210 + (((((int)clusterId) * 2560) + (((int)coreId) * 640)))), ((float *)input0_local_nram + (0)), 2560, NRAM2GDRAM);
    }
    """
    _ = pre_processing_pipeline(func_content, target="BANG")
