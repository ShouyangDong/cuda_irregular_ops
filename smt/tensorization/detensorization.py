import json

import detensorize_unittest
from pycparser import c_ast, c_generator, c_parser


class NodeTransformer(c_ast.NodeVisitor):
    def generic_visit(self, node):
        for field, old_value in iter_fields(node):
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, c_ast.Node):
                        value = self.visit(value)
                        if value is None:
                            continue
                        elif not isinstance(value, c_ast.Node):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, c_ast.Node):
                new_node = self.visit(old_value)
                setattr(node, field, new_node)
        return node


def iter_fields(node):
    # this doesn't look pretty because `pycparser` decided to have structure
    # for AST node classes different from stdlib ones
    index = 0
    children = node.children()
    while index < len(children):
        name, child = children[index]
        try:
            bracket_index = name.index("[")
        except ValueError:
            yield name, child
            index += 1
        else:
            name = name[:bracket_index]
            child = getattr(node, name)
            index += len(child)
            yield name, child


class Detensorizer(NodeTransformer):
    def __init__(self, func_defs):
        self.func_defs = func_defs
        self.parser = c_parser.CParser()
        self.parameter_mappings = {}

    def visit_FuncCall(self, node):
        if node.name.name in self.func_defs:
            func_def = self.func_defs[node.name.name]
            seq_def = self.parser.parse(func_def)
            if not isinstance(seq_def, c_ast.FileAST):
                raise ValueError("Sequential code must be a function")

            # Construct a map between the function call's  arguments and callee's arguments
            seq_def_args = seq_def.ext[0].decl.type.args.params
            seq_def_name = [arg_id.name for arg_id in seq_def_args]
            self.parameter_mappings = {
                arg: param for arg, param in zip(seq_def_name, node.args.exprs)
            }
            body = seq_def.ext[0].body
            return self.visit(body)
        else:
            return node

    def visit_ID(self, node):
        if node.name in self.parameter_mappings:
            return self.parameter_mappings[node.name]
        return node


def smt_transform(code, file_name):
    """
    Transform C code using an SMT solver to optimize loop constructs.

    This function parses the provided C code into an Abstract Syntax Tree (AST) and applies
    a transformation to split loops based on the given loop index and factor. The transformation
    is guided by an SMT solver to ensure the generated code is logically equivalent to the
    original but potentially more optimized.

    Parameters:
    - code (str): A string containing the C code to be transformed.
    - file_name (str): The definition of intrinsics.

    Returns:
    - str: The transformed C code as a string.

    Todo:
    - Implement additional error checking for the input parameters.
    - Extend the visitor to handle more complex loop structures.
    """
    # Parse the C code into an Abstract Syntax Tree (AST)
    # Assuming c_parser.CParser() is a valid parser for C code
    parser = c_parser.CParser()
    ast = parser.parse(code)

    with open(file_name) as json_file:
        func_defs = json.load(json_file)

    # Create an instance of a visitor that will perform the loop split
    # Assuming SplitForLoopVisitor is a class that knows how to split loops
    visitor = Detensorizer(func_defs)
    # Visit the AST with the visitor to apply the transformation
    visitor.visit(ast)

    # Generate the transformed code from the modified AST
    # Assuming c_generator.CGenerator() is a valid code generator for C
    generator = c_generator.CGenerator()
    transformed_code = generator.visit(ast)

    return transformed_code


class FuncCallsVisitor(NodeTransformer):
    def __init__(self):
        self.calles = []

    def visit_FuncCall(self, node):
        self.calles.append(node.name.name)


def prompt_generate(intrinsic, user_mannual):
    """
    Generate a prompt based on the intrinsic function and user manual.

    This function analyzes the given intrinsic function and user manual to determine
    the appropriate transformation or operation to be applied to the code. The result
    is a prompt that can guide the code transformation process.

    Parameters:
    - intrinsic (str): The name of the intrinsic function encountered in the code.
    - user_mannual (str): A manual or set of instructions provided by the user.

    Returns:
    - str: A prompt that indicates how to handle the intrinsic function based on the user manual.

    Raises:
    - ValueError: If the intrinsic function is not supported or the user manual is invalid.

    Todo:
    - Implement a more comprehensive analysis of the user manual to support various instructions.
    - Extend the function to handle a wider range of intrinsic functions.
    """
    # 检查内建函数是否受支持
    supported_intrinsics = {"__syncthreads", "__ballot", "__shfl"}
    if intrinsic not in supported_intrinsics:
        raise ValueError(f"Unsupported intrinsic function: {intrinsic}")

    # 根据用户手册生成提示
    # 这里只是一个简单的示例，实际情况可能需要解析更复杂的用户手册内容
    if user_mannual and "loop" in user_mannual.lower():
        prompt = f"Transform {intrinsic} into a loop construct."
    elif user_mannual and "atomic" in user_manical.lower():
        prompt = f"Ensure atomicity when handling {intrinsic}."
    else:
        # 默认提示
        prompt = f"Handle {intrinsic} with default transformation rules."

    return prompt


def gpt_transform(code, prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": question_system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return response["choices"][0]["message"]["content"]


def transform_block(code, user_mannual):
    """
    Apply a series of transformations to the input code based on user manual instructions.

    This function first transforms the code using an AI model (presumably 'gpt') that
    understands the user manual instructions. It then tests the transformed code with a
    unittest to check for any issues. If the unittest indicates a failure, the code
    is further refined using a Satisfiability Modulo Theories (SMT) solver to fix any
    potential problems.

    Parameters:
    - code (str): The original source code to be transformed.
    - user_mannual (str): Instructions provided by the user to guide the transformation.

    Returns:
    - str: The transformed and potentially fixed source code.

    Raises:
    - NotImplementedError: If any of the transformation functions are not implemented.

    Todo:
    - Add error handling for cases where transformation or testing fails.
    - Improve the unittest to cover more edge cases.
    """
    # traverse the intrinsic in the code
    parser = c_parser.CParser()
    ast = parser.parse(code)
    visitor = FuncCallsVisitor()
    visit.visit(ast)
    intrinsics = visitor.calles
    for intrinsic in intrinsics:
        # First transform the code using gpt
        prompt = prompt_generate(intrinsic, user_mannual)
        gpt_code = gpt_transform(code, prompt)
        # Test the code with unittest
        status = detensorize_unittest(gpt_code)
        # Fix the code with SMT
        if not status:
            code = smt_transform(code)
        else:
            code = gpt_code
    return code


if __name__ == "__main__":
    code = """
    void add_kernel0(float* lhs, float* rhs, float* add_1515) {
        float lhs_local_nram[128];
        if (((((int)clusterId) * 4) + ((int)coreId)) < 15) {
            __memcpy(((float *)lhs_local_nram + (0)), ((float *)lhs + (((((int)clusterId) * 256) + (((int)coreId) * 64)))), 256, GDRAM2NRAM);
        }
        if (((((int)clusterId) * 4) + ((int)coreId)) < 15) {
            __memcpy(((float *)lhs_local_nram + (64)), ((float *)rhs + (((((int)clusterId) * 256) + (((int)coreId) * 64)))), 256, GDRAM2NRAM);
        }
        if (((((int)clusterId) * 4) + ((int)coreId)) < 15) {
            __bang_add(((float *)lhs_local_nram + (0)), ((float *)lhs_local_nram + (0)), ((float *)lhs_local_nram + (64)), 64);
        }
        if (((((int)clusterId) * 4) + ((int)coreId)) < 15) {
            __memcpy(((float *)add_1515 + (((((int)clusterId) * 256) + (((int)coreId) * 64)))), ((float *)lhs_local_nram + (0)), 256, NRAM2GDRAM);
        }
    }
    """

    parser = c_parser.CParser()
    ast = parser.parse(code)
    v = FuncCallsRemover(
        file_name="/Users/dongshouyang/Downloads/micro/cuda_irregular_ops/function_definition.json"
    )
    v.visit(ast)
    generator = c_generator.CGenerator()
    print(generator.visit(ast))
