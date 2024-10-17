from src.post_processing.post_processing_prompt import (
    TENSORIZATION_PROMPT,
    TENSORIZATION_DEMO,
)


def tensorization(code, document):
    PROMPT = """
    {SYSTEM_PROMPT}
    
    Here is the introduction of Tensorization: {TENSORIZATION_PROMPT}
    Please transform the following code {code} accordingt to the demo:
    {TENSORIZATION_DEMO}"""

    PROMPT = PROMPT.replace("{SYSTEM_PROMPT}", SYSTEM_PROMPT)
    PROMPT = PROMPT.replace("{TENSORIZATION_PROMPT}", TENSORIZATION_PROMPT)
    PROMPT = PROMPT.replace("{TENSORIZATION_DEMO}", TENSORIZATION_DEMO)
    PROMPT = PROMPT.replace("{code}", code)
    return PROMPT


def get_operation_words(pragma_line):
    # Define the regex pattern to match all words inside parentheses after '#pragma operation'
    pattern = r"#pragma\s+operation\((\w+)\)"
    # Find all matches in the given string
    matches = re.findall(pattern, pragma_line)
    # Return the list of matches (it will be empty if no matches are found)
    return matches


def run_tensorization(code, target):
    op_list = get_operation_words(code)
    for op in op_list:
        op_document = op_dict[op]
        code = tensorization(code, op_document)
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
