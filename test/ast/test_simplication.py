from pycparser import c_ast, c_generator, c_parser

from smt.util import NodeTransformer


class SimplifyConstants(NodeTransformer):
    def visit_BinaryOp(self, node):
        # 检查是否是乘法操作
        if node.op == "*":
            # 如果两个操作数都是常量，则可以进行简化
            if isinstance(node.left, c_ast.Constant) and isinstance(
                node.right, c_ast.Constant
            ):
                # 计算并返回新的常量节点
                result = int(node.left.value) * int(node.right.value)
                return c_ast.Constant("int", value=str(result))
        elif node.op == "+":
            # 如果两个操作数都是常量，则可以进行简化
            if isinstance(node.left, c_ast.Constant) and isinstance(
                node.right, c_ast.Constant
            ):
                # 计算并返回新的常量节点
                result = int(node.left.value) + int(node.right.value)
                return c_ast.Constant("int", value=str(result))

        if node.op == "/":
            # 如果两个操作数都是常量，则可以进行简化
            if isinstance(node.left, c_ast.Constant) and isinstance(
                node.right, c_ast.Constant
            ):
                # 计算并返回新的常量节点
                result = int(node.left.value) / int(node.right.value)
                return c_ast.Constant("int", value=str(result))
        elif node.op == "-":
            # 如果两个操作数都是常量，则可以进行简化
            if isinstance(node.left, c_ast.Constant) and isinstance(
                node.right, c_ast.Constant
            ):
                # 计算并返回新的常量节点
                result = int(node.left.value) - int(node.right.value)
                return c_ast.Constant("int", value=str(result))

        else:
            return self.generic_visit(node)


def simplify_code(source_code):
    # 解析 C 代码
    parser = c_parser.CParser()
    ast = parser.parse(source_code)
    generator = c_generator.CGenerator()
    # 创建自定义访问器实例
    visitor = SimplifyConstants()
    # 访问 AST 以进行常量折叠
    visitor.visit(ast)
    # 生成简化后的 C 代码
    return generator.visit(ast)


# 使用示例
source = """ 
void add(int* a, int* b) {
    for (int i_j_fuse = 0; i_j_fuse < 300 * 300; i_j_fuse++) {
        a[i_j_fuse] = b[i_j_fuse] + 4;
    }
}
"""

simplified_source = simplify_code(source)
print(simplified_source)
