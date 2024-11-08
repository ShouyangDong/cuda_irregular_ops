import re

import openai

from falcon.src.loop_transformation.decorate_pragma import SPLIT_PRAGMA_PROMPT
from falcon.src.prompt.prompt import SYSTEM_PROMPT

model_name = """gpt-4-turbo"""
openai.api_key = "sk-JmlwEmWiNtFqSD7IDaF981Dd8a7447FfBcE768755cB38010"
openai.api_base = "https://api.keya.pw/v1"


def run_split_annotation(code):
    PROMPT = """
    {SYSTEM_PROMPT}
    {SPLIT_PRAGMA_PROMPT}
    """
    PROMPT = PROMPT.replace("{SYSTEM_PROMPT}", SYSTEM_PROMPT)

    PROMPT = PROMPT.replace("{SPLIT_PRAGMA_PROMPT}", SPLIT_PRAGMA_PROMPT)
    PROMPT = PROMPT.replace("{code}", code)
    transformation_completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "user", "content": PROMPT}],
    )

    content = transformation_completion.choices[0].message["content"]
    match = re.search(r'```[a-zA-Z]*\n(.*?)```', content, re.S)
    if match:
        code_content = match.group(1).strip()
        return code_content
    return None


if __name__ == "__main__":
    code = """
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
    output_code = run_split_annotation(code)
    print(output_code)

    code = """
    extern "C" void  add_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ T_add) {
        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 1024; j++) {
                    T_add[((i * 1024) + j)] = (A[((i * 1024) + j)] + B[((i * 1024) + j)]);
                }
            }
        }
    }
    """
    code = run_split_annotation(code)
    print(code)
