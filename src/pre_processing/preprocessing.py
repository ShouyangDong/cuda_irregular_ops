# autoflake: skip_file
import openai

from src.pre_processing.preprocessing_prompt import *


OPT_LIST = ["LOOP_RECOVERY", "DETENSORIZATION"]
openai.api_key = """ OPENAI API KEY """


def pre_processing_pipeline(func_content, target):
    """This function transforms the given code by performing two main transformations:
        1. Convert parallel loop variables (e.g., OpenMP, CUDA) into standard C for loops.
        2. Convert SIMD tensor operations into scalar for-loop based calculations.
    :param func_content: The content of the function (code) to be transformed.
    :return: Transformed code after applying the two transformations."""
    TRANS_DESCRIPTION = ""
    for trans in OPT_LIST:
        prompt_name = (
            f"{trans}_PROMPT_{target}"
            if trans != "DETENSORIZATION"
            else f"{trans}_PROMPT"
        )
        prompt_content = globals()[prompt_name]
        TRANS_DESCRIPTION += prompt_content
    return TRANS_DESCRIPTION


if __name__ == "__main__":
    func_content = """
    extern "C" __mlu_global__ void tanh(float* input0, float* active_tanh_210) {
        __nram__ float input0_local_nram[640];
        __memcpy(((float *)input0_local_nram + (0)), ((float *)input0 + (((((int)clusterId) * 2560) + (((int)coreId) * 640)))), 2560, GDRAM2NRAM);
        __bang_active_tanh(((float *)input0_local_nram + (0)), ((float *)input0_local_nram + (0)), 640);
        __memcpy(((float *)active_tanh_210 + (((((int)clusterId) * 2560) + (((int)coreId) * 640)))), ((float *)input0_local_nram + (0)), 2560, NRAM2GDRAM);
    }
    """
    pre_processing_pipeline(func_content, target="BANG")
