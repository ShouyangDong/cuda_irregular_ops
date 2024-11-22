import random

from pycparser import c_ast, c_generator, c_parser


class LoopReorderVisitor(c_ast.NodeVisitor):
    def __init__(self):
        self.for_loops = []

    def visit_For(self, node):
        self.for_loops.append(node)
        self.generic_visit(node)

    def reorder_loops(self):
        # 遍历所有的 for 循环，寻找可以交换的嵌套循环
        for node in self.for_loops:
            if isinstance(node.stmt, c_ast.Compound) and node.stmt.block_items:
                first_item = node.stmt.block_items[0]
                if isinstance(first_item, c_ast.For):
                    # 随机决定是否交换
                    if random.choice([True, False]):
                        inner_loop = first_item
                        stmt_node = inner_loop.stmt
                        # 交换外层和内层循环
                        new_inner_loop = c_ast.For(
                            init=node.init,
                            cond=node.cond,
                            next=node.next,
                            stmt=stmt_node,
                        )
                        node.init = inner_loop.init
                        node.cond = inner_loop.cond
                        node.next = inner_loop.next
                        node.stmt = c_ast.Compound(block_items=[new_inner_loop])


def ast_loop_reorder(c_code):
    # 解析 C 代码
    parser = c_parser.CParser()
    ast = parser.parse(c_code)
    generator = c_generator.CGenerator()
    # 创建访问者实例并进行循环交换
    visitor = LoopReorderVisitor()
    visitor.visit(ast)
    visitor.reorder_loops()
    return generator.visit(ast)


if __name__ == "__main__":
    c_code = """
    int factorial(int result) {
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                result += i + j;
            }
        }
        return result;
    }
    """
    code = ast_loop_reorder(c_code)
    print(code)
