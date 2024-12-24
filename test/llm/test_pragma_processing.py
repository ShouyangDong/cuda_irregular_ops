import re


def get_operation_content(code):
    # Define the regex pattern to match the content inside parentheses
    # following #pragma intrinsic
    pattern = r"#pragma\s+operation\(([^)]+)\)"
    # Find all matches in the given code (there might be multiple pragmas)
    matches = re.findall(pattern, code)
    return matches


def get_input_operand(pragma):
    inputs = pragma.split("input[")[1].split("]")[0]
    input_list = inputs.split(", ")
    return input_list


def get_output_operand(pragma):
    outputs = pragma.split("output[")[1].split("]")[0]
    output_list = outputs.split(", ")
    return output_list


def replace_operation_with_intrinsic(code, op_pragma):
    # Get the list of operations from the code
    op_list = get_operation_content(code)
    space_maps = []
    # Iterate over each operation found in the code
    for op in op_list:
        op_name = op.split("(input")[0]
        # Build the pragma search pattern to match the specific operation
        # pragma
        pragma_pattern = re.escape(f"#pragma operation({op})")
        # Ensure the operation exists in the op_pragma dictionary
        if op_name not in op_pragma:
            raise KeyError(f"Operation '{op_name}' not found in op_pragma.")
        # Get input and output operands for the operation
        input_operands = get_input_operand(op)
        output_operands = get_output_operand(op)
        # Get corresponding memory spaces from the op_pragma dictionary
        input_spaces = get_input_operand(op_pragma[op_name])
        output_spaces = get_output_operand(op_pragma[op_name])
        # Ensure the input/output operand lists and spaces have matching
        # lengths
        if len(input_operands) != len(input_spaces):
            raise ValueError(
                f"Input operands and memory spaces length mismatch for operation '{op_name}' "
                f"({len(input_operands)} operands vs {len(input_spaces)} spaces)."
            )

        if len(output_operands) != len(output_spaces):
            raise ValueError(
                f"Output operands and memory spaces length mismatch for operation '{op_name}' "
                f"({len(output_operands)} operands vs {len(output_spaces)} spaces)."
            )
        # Create the space map by zipping operands and spaces
        input_map = {
            operand: space
            for operand, space in zip(input_operands, input_spaces)
        }
        output_map = {
            operand: space
            for operand, space in zip(output_operands, output_spaces)
        }
        space_map = {"input": input_map, "output": output_map}
        # Replace the matching pragma with the corresponding value from
        # op_pragma
        code = re.sub(pragma_pattern, op_pragma[op_name], code)
        # Append the space map for this operation
        space_maps.append(space_map)
    return code, space_maps


if __name__ == "__main__":
    code = """
    #pragma operation(matmul(input[A, B], output[C]))
    for (int col = 0; col < 64; col++) {
        C[(clusterId * 4 + coreId) * 64 + col] = 0.0f;
        for (int i = 0; i < 512; i++) {
            C[col] += A[i] * B[i * 64 + col];
        }
    }
    #pragma operation(add(input[A, B], output[C]))
    for (int i = 0; i < 512; i++) {
        C[i] = A[i] + B[i];
    }
    """
    op_pragma = {
        "matmul": "#pragma intrinsic(__bang_mlp(input[Nram, Wram], output[Nram]))",
        "add": "#pragma intrinsic(__bang_add(input[Nram, Nram], output[Nram]))",
    }
    final_code, space_maps = replace_operation_with_intrinsic(code, op_pragma)
    print(final_code)
    print(space_maps)
