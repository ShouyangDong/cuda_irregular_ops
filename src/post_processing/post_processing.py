import re
import openai

from src.post_processing.post_processing_prompt import *
from src.prompt.prompt import SYSTEM_PROMPT, PRAGMA_INSERT_PROMPT, APPLY_OPT_PROMPT

TENSORIZATION_OPT_LIST = [
    "CACHE_READ",
    "CACHE_WRITE",
    "TENSORIZATION",
]

OPT_PROCESSURE = [
    "THREAD_BINDING",
    "TENSORIZATION",
    "DOUBLE_BUFFER",
]

model_name = """gpt-3.5-turbo"""
openai.api_key = "sk-JmlwEmWiNtFqSD7IDaF981Dd8a7447FfBcE768755cB38010"
openai.api_base = "https://api.keya.pw/v1"

POST_PROCESSING_PRAGMA_HINTS = {
    "THREAD_BINDING": "#pragma thread_bind({target})",
    "TENSORIZATION": "#pragma",
    "DOUBLE_BUFFER": "#pragma double_buffer({target})",
}


def run_code_analysis(code, pass_name, target):
    PRAGMA_DESCRIPTION = globals()[pass_name + "_PROMPT_" + target]
    _PRAGMA_INSERT_PROMPT = PRAGMA_INSERT_PROMPT.replace(
        "{PRAGMA_DESCRIPTION}", PRAGMA_DESCRIPTION
    )
    _PRAGMA_INSERT_PROMPT = _PRAGMA_INSERT_PROMPT.replace(
        "{PRAGMA_NAME}",
        POST_PROCESSING_PRAGMA_HINTS[pass_name].replace("{target}", target),
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


def run_code_transformation(code, pass_name, target):
    if pass_name != "THREAD_BINDING":
        # Traverse all possible pragma, just follow the tensorization schedule
        for pass_name in TENSORIZATION_OPT_LIST:
            code = run_code_transformation(code, pass_name, target)
        return code

    PRAGMA_DEMO_COMPLETE = globals()[pass_name + "_DEMO_" + target]
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


def post_processing_pipeline(func_content, target):
    """This function transforms the given code by performing two main transformations:
        1. Convert parallel loop variables (e.g., OpenMP, CUDA) into standard C for loops.
        2. Convert SIMD tensor operations into scalar for-loop based calculations.
    :param func_content: The content of the function (code) to be transformed.
    :return: Transformed code after applying the two transformations."""
    for i, trans in enumerate(OPT_PROCESSURE):
        # First analysis the code, and insert corresponding tensorize pragma
        func_content = run_code_analysis(func_content, trans, target)
        # Transform the code according to the pragma
        func_content = run_code_transformation(func_content, trans, target)

    return func_content


if __name__ == "__main__":
    func_content = """
    extern "C" void add_kernel(float* output, float* input1, float* input2) {
        int dim1 = 4;
        int dim2 = 4;
        int dim3 = 4;
        int dim4 = 64;
        
        for (int i = 0; i < dim1; i++) {
            for (int j = 0; j < dim2; j++) {
                for (int k = 0; k < dim3; k++) {
                    for (int l = 0; l < dim4; l++) {
                        int index = i * dim2 * dim3 * dim4 + j * dim3 * dim4 + k * dim4 + l;
                        output[index] = input1[index] + input2[index];
                    }
                }
            }
        }
    }
    """
    _ = post_processing_pipeline(func_content, "BANG")
