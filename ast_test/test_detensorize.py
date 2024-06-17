from pycparser import c_parser, c_ast, c_generator
import json

class FuncCallVisitor(c_ast.NodeVisitor):
    def __init__(self, file_name):
        with open(file_name) as json_file:
            self.func_defs = json.load(json_file)
        self.parser = c_parser.CParser()

    def visit_FuncCall(self, node):
        if node.name.name in self.func_defs:
            func_def = self.func_defs[node.name.name]
            seq_def = self.parser.parse(func_def)

            if not isinstance(seq_def, c_ast.FileAST):
                raise ValueError("Sequential code must be a function")
            
            # Construct a map between the function call's  arguments and callee's arguments
            seq_def_args = seq_def.ext[0].decl.type.args.params
            parameter_mappings = {arg: param for arg, param in zip(node.args.exprs, seq_def_args)}
            # 替换函数调用节点
            new_body = self.replace(node, seq_def.ext[0].body, parameter_mappings)
            node.parent.body.remove(node)  # 从父节点中移除原函数调用节点
            node.parent.body.extend(new_body)  # 将新节点列表添加到父节点

    def replace(self, node, new_body, mappings):
        """
        使用新体替换函数调用节点，并更新参数映射。
        """
        # 创建新的参数列表和新的函数调用列表
        new_params = [c_ast.Decl(decl.name, c_ast.TypeDecl()) for decl in new_body.ext[0].decls()]
        new_func_calls = [c_ast.FuncCall(c_ast.ID(new_params[i].name), []) for i in range(len(new_params))]

        # 更新参数映射，将新参数映射到原函数调用的参数
        for old_arg, new_arg in zip(node.args.exprs, new_func_calls):
            mappings[old_arg] = new_arg

        # 替换函数调用节点的参数
        for old_arg, new_arg in mappings.items():
            new_body = new_body.replace(old_arg, new_arg)
        return new_body

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
