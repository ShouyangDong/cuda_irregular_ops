import re
import openai

from src.prompt.prompt import SYSTEM_PROMPT
from src.pre_processing.preprocessing_prompt import (
    LOOP_RECOVERY_PROMPT_CUDA,
    LOOP_RECOVERY_DEMO_CUDA,
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
    PROMPT = PROMPT.replace("{TENSORIZATION_PROMPT}", LOOP_RECOVERY_PROMPT_CUDA)
    PROMPT = PROMPT.replace("{LOOP_RECOVERY_DEMO}", LOOP_RECOVERY_DEMO_CUDA)
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


if __name__ == "__main__":
    code = """
    extern "C" __global__ void __launch_bounds__(1024) add_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ T_add) {
        if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 2309) {
            T_add[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (A[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] + B[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
        }
    }
    """
    code = run_loop_recovery(code, target="CUDA")
    print(code)
