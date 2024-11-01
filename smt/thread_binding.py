from pycparser import c_ast, c_generator, c_parser

from smt.util import NodeTransformer

builtin_var = {"CUDA": ["threadIdxx", "blockIdxx"], "BANG": ["coreId", "clusterId"]}


class ThreadBindingTransformer(NodeTransformer):
    def __init__(self, parallel_loops, target="BANG"):
        self.binding_map = {}
        self.loop_order = 0
        self.parallel_loops = parallel_loops
        self.target = target

    def visit_For(self, node):
        # 检查循环变量是否是绑定变量，如果是则返回循环体
        loop_var = (
            node.init.decls[0].name if isinstance(node.init, c_ast.DeclList) else None
        )
        if self.parallel_loops >= 2:
            # 记录并行循环的层次
            if self.parallel_loops >= 3:
                if self.loop_order == 0:
                    self.binding_map[loop_var] = builtin_var[self.target][1]
                elif self.loop_order == 1:
                    self.binding_map[loop_var] = builtin_var[self.target][0]

            elif self.parallel_loops == 2:
                if self.loop_order == 0:
                    # 两层并行循环，绑定 coreId
                    self.binding_map[loop_var] = builtin_var[self.target][0]
            self.loop_order += 1
            node = self.generic_visit(node)
            self.loop_order -= 1
        else:
            node = self.generic_visit(node)

        if node.init.decls[0].name in self.binding_map:
            return node.stmt.block_items[0]
        return node

    def visit_ID(self, node):
        if node.name in self.binding_map:
            node.name = self.binding_map[node.name]
        return node


class LoopVisitor(c_ast.NodeVisitor):
    def __init__(self):
        self.parallel_loops = 0

    def visit_For(self, node):
        self.parallel_loops += 1
        self.generic_visit(node)


def ast_thread_binding(code, target="BANG"):
    # 解析代码
    parser = c_parser.CParser()
    ast = parser.parse(code)

    # 统计循环层数
    loop_visitor = LoopVisitor()
    loop_visitor.visit(ast)

    # 进行线程绑定转换
    transformer = ThreadBindingTransformer(loop_visitor.parallel_loops, target)
    ast = transformer.visit(ast)

    # 输出修改后的代码
    generator = c_generator.CGenerator()
    return generator.visit(ast)


if __name__ == "__main__":
    # 示例代码
    code = """
    void func() {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                for (int k = 0; k < 7; ++k) {
                    B[i * 4 * 7 + j * 7 + k] = A[i * 4 * 7 + j * 7 + k] + 1.0;
                }
            }
        }
    }
    """
    output_code = ast_thread_binding(code, target="BANG")
    print(output_code)

    code = """
    void func() {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                for (int k = 0; k < 7; ++k) {
                    B[i * 4 * 7 + j * 7 + k] = A[i * 4 * 7 + j * 7 + k] + 1.0;
                }
            }
        }
    }
    """
    output_code = ast_thread_binding(code, target="CUDA")
    print(output_code)
