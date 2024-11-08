import json

from pycparser import c_ast, c_generator, c_parser

from falcon.simplification import simplify_code
from falcon.smt.util import NodeTransformer, remove_target_prefix
from falcon.stmt_simplification import ast_stmt_simplification
from falcon.buffer_inline import ast_buffer_inline

file_name = "falcon/documents/op_tensorization.json"


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
            return self.visit(body.block_items[0])
        else:
            return node

    def visit_ID(self, node):
        if node.name in self.parameter_mappings:
            return self.parameter_mappings[node.name]
        return node


def ast_detensorization(code, target):
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
    code = remove_target_prefix(code, target)
    if target == "BANG":
        parser = c_parser.CParser()
        ast = parser.parse(code)
        with open(file_name) as json_file:
            func_defs = json.load(json_file)
        visitor = Detensorizer(func_defs)
        visitor.visit(ast)
        generator = c_generator.CGenerator()
        code = generator.visit(ast)

    code = simplify_code(code)
    code = ast_stmt_simplification(code)
    code = ast_buffer_inline(code)
    return code


if __name__ == "__main__":
    code = """
    void add(float* lhs, float* rhs, float* add_1515) {
        float lhs_local_nram[128];
        __memcpy(((float *)lhs_local_nram + (0)), ((float *)lhs + (((((int)clusterId) * 256) + (((int)coreId) * 64)))), 256, GDRAM2NRAM);
        __memcpy(((float *)lhs_local_nram + (64)), ((float *)rhs + (((((int)clusterId) * 256) + (((int)coreId) * 64)))), 256, GDRAM2NRAM);
        __bang_add(((float *)lhs_local_nram + (0)), ((float *)lhs_local_nram + (0)), ((float *)lhs_local_nram + (64)), 64);
        __memcpy(((float *)add_1515 + (((((int)clusterId) * 256) + (((int)coreId) * 64)))), ((float *)lhs_local_nram + (0)), 256, NRAM2GDRAM);
    }
    """
    code = ast_detensorization(code, "BANG")
    print(code)
    code = """
        void tanh(float* input0, float* active_tanh_210) {
        float input0_local_nram[640];
        __memcpy(((float *)input0_local_nram + (0)), ((float *)input0 + (((((int)clusterId) * 2560) + (((int)coreId) * 640)))), 2560, GDRAM2NRAM);
        __bang_active_tanh(((float *)input0_local_nram + (0)), ((float *)input0_local_nram + (0)), 640);
        __memcpy(((float *)active_tanh_210 + (((((int)clusterId) * 2560) + (((int)coreId) * 640)))), ((float *)input0_local_nram + (0)), 2560, NRAM2GDRAM);
    }
    """
    code = ast_detensorization(code, "BANG")
    print(code)
