from pycparser import c_ast, c_generator, c_parser

from smt.simplification import simplify_code
from smt.util import NodeTransformer


class FuncCallsRemover(NodeTransformer):
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
    func_map = {
        "__memcpy": "void memcpy(float* dst, float* src, int size, char direction) {for(int i=0; i<size/4; i++) {dst[i]=src[i];}}",
        "__bang_add": "void bang_add(float* C, float* A, float* B, int size) {for(int i=0; i < size; i++){C[i]=A[i] + B[i];}}",
    }

    parser = c_parser.CParser()
    ast = parser.parse(code)
    v = FuncCallsRemover(func_map)
    v.visit(ast)
    generator = c_generator.CGenerator()
    code = simplify_code(generator.visit(ast))
    print(code)
