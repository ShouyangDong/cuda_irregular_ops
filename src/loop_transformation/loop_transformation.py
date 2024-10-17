import re
import openai

from src.loop_transformation.loop_transformation import *
from src.prompt.prompt import SYSTEM_PROMPT, PRAGMA_INSERT_PROMPT, APPLY_OPT_PROMPT

opt_options = [
    "LOOP_FUSION",
    "LOOP_REORDER",
    "LOOP_SPLIT",
    "TENSOR_COMTRACTION",
]

model_name = """gpt-3.5-turbo"""
openai.api_key = "sk-JmlwEmWiNtFqSD7IDaF981Dd8a7447FfBcE768755cB38010"
openai.api_base = "https://api.keya.pw/v1"


LOOP_TRANSFORMATION_HINTS = {
    "LOOP_FUSION": "#pragma loop_fusion",
    "LOOP_REORDER": "#pragma loop_reorder",
    "LOOP_SPLIT": "#pragma loop_split(factor={factor})",
    "TENSOR_COMTRACTION": "#pragma tensor_contraction"
}


def run_code_analysis(code, pass_name):
    PRAGMA_DESCRIPTION = globals()[pass_name + "_PROMPT"]
    _PRAGMA_INSERT_PROMPT = PRAGMA_INSERT_PROMPT.replace(
        "{PRAGMA_DESCRIPTION}", PRAGMA_DESCRIPTION
    )
    _PRAGMA_INSERT_PROMPT = _PRAGMA_INSERT_PROMPT.replace(
        "{PRAGMA_NAME}",
        PRE_PROCESSING_PRAGMA_HINTS[pass_name],
    )
    _PRAGMA_INSERT_PROMPT = _PRAGMA_INSERT_PROMPT.replace("{STAGE_CODE_CONTENT}", code)

    STAGE_OPT_PROMPT_COMPLETE = SYSTEM_PROMPT + _PRAGMA_INSERT_PROMPT
    analysis_completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "user", "content": STAGE_OPT_PROMPT_COMPLETE}],
    )
    content = analysis_completion.choices[0].message["content"]

    match = re.search(r"\`\`\`(.*?)\`\`\`", content, re.DOTALL)
    print("[INFO]*********final code: ", content)
    return match


def run_code_transformation(code, pass_name):
    PRAGMA_DEMO_COMPLETE = globals()[pass_name + "_DEMO"]
    _APPLY_OPT_PROMPT = APPLY_OPT_PROMPT.replace("{STAGE_CODE_CONTENT}", func_content)
    _APPLY_OPT_PROMPT = _APPLY_OPT_PROMPT.replace("{OPT_LIST}", pass_name)
    _APPLY_OPT_PROMPT = _APPLY_OPT_PROMPT.replace("{PRAGMA_DEMO}", PRAGMA_DEMO_COMPLETE)

    STAGE_OPT_PROMPT_COMPLETE = SYSTEM_PROMPT + _APPLY_OPT_PROMPT

    transformation_completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "user", "content": STAGE_OPT_PROMPT_COMPLETE}],
    )
    content = transformation_completion.choices[0].message["content"]
    match = re.search(r"\`\`\`(.*?)\`\`\`", content, re.DOTALL)
    print("[INFO]*********transformation: ", match)
    return match


def loop_transformation_pipeline(func_content, target):
    """This function transforms the given code by performing some loop transformations"""
    for i, trans in enumerate(OPT_LIST):
        # First analysis the code, and insert corresponding pragma
        func_content = run_code_analysis(func_content, trans)
        # Transform the code according to the pragma
        func_content = run_code_transformation(func_content, trans)
    return func_content


if __name__ == "__main__":
    func_content = """
    extern "C" __mlu_global__ void tanh(float* input0, float* active_tanh_210) {
        __nram__ float input0_local_nram[640];
        __memcpy(((float *)input0_local_nram + (0)), ((float *)input0 + (((((int)clusterId) * 2560) + (((int)coreId) * 640)))), 2560, GDRAM2NRAM);
        __bang_active_tanh(((float *)input0_local_nram + (0)), ((float *)input0_local_nram + (0)), 640);
        __memcpy(((float *)active_tanh_210 + (((((int)clusterId) * 2560) + (((int)coreId) * 640)))), ((float *)input0_local_nram + (0)), 2560, NRAM2GDRAM);
    }
    """
    _ = loop_transformation_pipeline(func_content, target="BANG")
