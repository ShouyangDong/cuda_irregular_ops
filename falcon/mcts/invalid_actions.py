from pycparser import c_ast, c_parser

from falcon.mcts.actions import actions as ActionSpace
from falcon.util import remove_target_prefix


class NodeTransformer(c_ast.NodeVisitor):
    def __init__(self):
        self.func_call = False

    def visit_FuncCall(self, node):
        self.func_call = True


def visit_func_call(code):
    code = remove_target_prefix(code)
    # 解析代码
    parser = c_parser.CParser()
    ast = parser.parse(code)

    # 统计循环层数
    loop_visitor = NodeTransformer()
    loop_visitor.visit(ast)
    return loop_visitor.func_call


def get_invalid_actions(code, source_platform, target_platform):
    invalid_mask = [0] * len(ActionSpace)

    if source_platform == "cpu":
        invalid_mask[0] = 1

    if not visit_func_call(code):
        invalid_mask[2] = 1

    if target_platform == "cpu":
        invalid_mask[7] = 1
        invalid_mask[8] = 1
        invalid_mask[10] = 1

    if "coreId" not in code or "threadIdx." not in code:
        invalid_mask[0] = 1
    return invalid_mask


if __name__ == "__main__":
    code = """
    int square(int x) {
        return x * x;
    }
    """
    result = visit_func_call(code)
    print(result)

    code = """
    int main() {
        int a = 3;
        int b = square(a);  // <--- 函数调用
        return b;
    }
    """
    result = visit_func_call(code)
    print(result)

    code = """
    int main() {
        int a = 3;
        square(a);  // <--- 函数调用
        return a;
    }
    """
    result = visit_func_call(code)
    print(result)
