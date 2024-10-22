import re
import openai
import json
from src.prompt.prompt import SYSTEM_PROMPT
from src.post_processing.post_processing_prompt import (
    TENSORIZATION_PROMPT,
)

model_name = """gpt-3.5-turbo"""
openai.api_key = "sk-JmlwEmWiNtFqSD7IDaF981Dd8a7447FfBcE768755cB38010"
openai.api_base = "https://api.keya.pw/v1"


def tensorization(op, code, document):
    PROMPT = """
    {SYSTEM_PROMPT}
    
    Here is the introduction of Tensorization: {TENSORIZATION_PROMPT}
    Please tensorize the sequential code of {op} below the #pragma operation in {code} 
    accordingt to the introduction of tensorized intrinsic.
    {document}
    Please return the output kernel function without any additional information.
    """

    PROMPT = PROMPT.replace("{SYSTEM_PROMPT}", SYSTEM_PROMPT)
    PROMPT = PROMPT.replace("{TENSORIZATION_PROMPT}", TENSORIZATION_PROMPT)
    PROMPT = PROMPT.replace("{document}", document)
    PROMPT = PROMPT.replace("{code}", code)
    PROMPT = PROMPT.replace("{op}", op)

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


def get_operation_words(pragma_line):
    # Modify the pattern to capture everything inside parentheses, allowing spaces and underscores
    pattern = r"#pragma\s+operation\(([^)]+)\)"
    # Find all matches in the given string
    matches = re.findall(pattern, pragma_line)
    # Return the list of matches (it will be empty if no matches are found)
    return matches


def run_tensorization(code, target):
    op_dict = json.load(open("./documents/bang_c_user_guide", "r"))
    op_list = get_operation_words(code)
    for op in op_list:
        op = "memcpy" if "memory" in op else op
        op_document = op_dict[op]
        code = tensorization(op, code, op_document)
    return code


if __name__ == "__main__":
    code = """
    #pragma operation(memory load into B_wram)
    for (int col = 0; col < 64; col++) {
        for (int i = 0; i < 512; i++) {
            B_wram[i * 64 + col] = B[i * 64 + col];

    #pragma operation(memory load into A_nram)
    for (int i = 0; i < 512; i++) {
        A_nram[i] = A[(clusterId * 4 + coreId) * 512 + i];
    }


    #pragma operation(matmul)
    for (int col = 0; col < 64; col++) {
        C_nram[(clusterId * 4 + coreId) * 64 + col] = 0.0f;
        for (int i = 0; i < 512; i++) {
            C_nram[col] += A_nram[i] * B_wram[i * 64 + col];
        }
    }

    #pragma operation(memory store to C)
    for (int col = 0; col < 64; col++) {
        C[(clusterId * 4 + coreId) * 64 + col] = C_nram[col];
    }
    """
    code = run_tensorization(code, target="BANG")
