from pycparser import c_parser, c_ast, c_generator
import json

class FuncCallVisitor(c_ast.NodeVisitor):
    def __init__(self, file_name):
        with open(file_name) as json_file:
            self.func_defs = json.load(json_file)
        self.parser = c_parser.CParser()

    def visit_FuncCall(self, node):
        if node.name.name == "__memcpy" or "__bang" in node.name.name:
            func_def = self.func_defs[node.name.name]
            seq_def = self.parser.parse(func_def)

            # Ensure the sequential code is a function
            assert isinstance(seq_def, c_ast.FileAST), "Sequential code  must be a function"
            
            # Construct a map between the function call's  arguments and callee's arguments
            seq_def_arg = seq_def.ext[0].decl.type.args.params
            parameter_mappings = {}
            for key, value in zip(node.args.exprs, seq_def_arg):
                parameter_mappings[key] = value

            # print(parameter_mappings)
            seq_def_body = seq_def.ext[0].body




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
    v = FuncCallVisitor(file_name = "/Users/dongshouyang/Downloads/micro/cuda_irregular_ops/function_definition.json")
    v.visit(ast)

    generator = c_generator.CGenerator()
    print(generator.visit(ast))
