import re


def get_operation_words(pragma_line):
    # Define the regex pattern to match all words inside parentheses after '#pragma operation'
    pattern = r"#pragma\s+operation\((\w+)\)"
    # Find all matches in the given string
    matches = re.findall(pattern, pragma_line)
    # Return the list of matches (it will be empty if no matches are found)
    return matches


def replace_pragma_with_intrinsic(code, op_pragma):
    # Get the list of operations from the code
    op_list = get_operation_words(code)
    # Iterate over each operation found in the code
    for op in op_list:
        # Build the pragma search pattern to match the specific operation pragma
        pragma_pattern = f"#pragma\s+operation\({op}\)"
        # Check if the operation exists in the op_pragma dictionary
        if op in op_pragma:
            # Replace the matching pragma with the corresponding value from op_pragma
            code = re.sub(pragma_pattern, op_pragma[op], code)
            # Output the modified code
    return code


if __name__ == "__main__":
    code = """
    #pragma operation(matmul)
    for (int col = 0; col < 64; col++) {
        C[(clusterId * 4 + coreId) * 64 + col] = 0.0f;
        for (int i = 0; i < 512; i++) {
            C[col] += A[i] * B[i * 64 + col];
        }
    }
    #pragma operation(add)
    for (int i = 0; i < 512; i++) {
        C[i] = A[i] + B[i];
    }
    """
    op_pragma = {
        "matmul": "#pragma intrinsic(__bang_mlp(input[Nram, Wram], output[Nram]))",
        "add": "#pragma intrinsic(__bang_add(input[Nram], output[Nram]))",
    }
    final_code = replace_pragma_with_intrinsic(code, op_pragma)
