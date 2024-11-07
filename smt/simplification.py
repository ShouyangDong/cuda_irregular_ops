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
                result = int(node.left.value) // int(node.right.value)
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

    def visit_If(self, node):
        if (
            isinstance(node.cond, c_ast.BinaryOp)
            and node.cond.op == "<"
            and node.cond.right.value == "4"
            and (node.cond.left.name == "coreId" or node.cond.left.name == "clusterId")
        ):
            return self.generic_visit(node.iftrue.block_items[0])
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


if __name__ == "__main__":
    c_code = """
    int factorial(int result) {
        if(coreId < 2) {
            for (int j = 0; j < 10; j++) {
                result += j;
            }
        }
        return result;
    }
    """
    code = simplify_code(c_code)
    print(code)
    c_code = """
    int factorial(int result) {
        if(clusterId < 4) {
            for (int j = 0; j < 10; j++) {
                result += j;
            }
        }
        return result;
    }
    """
    code = simplify_code(c_code)
    print(code)
