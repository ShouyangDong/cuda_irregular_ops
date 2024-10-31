from pycparser import c_ast, c_generator, c_parser

from smt.util import NodeTransformer


class ThreadBindingTransformer(NodeTransformer):
    def visit_For(self, node):
        # 检查是否是三重嵌套的 for 循环结构
        if isinstance(node.stmt, c_ast.Compound) and len(node.stmt.block_items) == 1:
            inner_for1 = node.stmt.block_items[0]
            if isinstance(inner_for1, c_ast.For) and isinstance(
                inner_for1.stmt, c_ast.Compound
            ):
                inner_for2 = inner_for1.stmt.block_items[0]
                if isinstance(inner_for2, c_ast.For):
                    # 开始转换嵌套的循环结构
                    return self.transform_loops(node, inner_for1, inner_for2)
        return node  # 如果不符合三层嵌套结构，则返回原节点

    def transform_loops(self, outer_for, middle_for, inner_for):
        """将外层和中间层的循环变量替换为 clusterId 和 coreId，并去除这些循环"""
        # 提取循环变量名
        outer_var = outer_for.init.decls[0].name  # 'i'
        middle_var = middle_for.init.decls[0].name  # 'j'

        # 使用 clusterId 和 coreId 替换外层和中间层的循环变量
        self.replace_var_with_thread_id(inner_for, outer_var, "clusterId")
        self.replace_var_with_thread_id(inner_for, middle_var, "coreId")

        # 返回只有内层循环的 Compound 结构
        return c_ast.Compound(block_items=[inner_for])

    def replace_var_with_thread_id(self, node, var_name, thread_id):
        """递归地将变量替换为指定的线程绑定标识符"""
        for _, child in node.children():
            if isinstance(child, c_ast.ID) and child.name == var_name:
                child.name = thread_id  # 替换变量名
            else:
                self.replace_var_with_thread_id(child, var_name, thread_id)


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

# 解析代码
parser = c_parser.CParser()
ast = parser.parse(code)

# 进行线程绑定转换
transformer = ThreadBindingTransformer()
ast = transformer.visit(ast)

# 输出修改后的代码
generator = c_generator.CGenerator()
output_code = generator.visit(ast)
print(output_code)
