import re
from pycparser import c_ast, c_generator, c_parser

from falcon.smt.util import NodeTransformer


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
    print(source_code)
    # 移除所有 C/C++ 样式的注释
    source_code = re.sub(r"//.*?\n|/\*.*?\*/", "", source_code, flags=re.S)
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

    c_code = """
    void sign(float *input0, float *active_sign_147)
    {
        for (int clusterId = 0; clusterId < 4; ++clusterId)
        {
            for (int coreId = 0; coreId < 4; ++coreId)
            {
                float input0_local_nram[25];
                for (int i0_outer_outer_outer = 0; i0_outer_outer_outer < 3; ++i0_outer_outer_outer)
                {
                    if ((((i0_outer_outer_outer * 16) + (((int) clusterId) * 4)) + ((int) coreId)) < 45)
                    {
                        int src_offset = ((i0_outer_outer_outer * 400) + (((int) clusterId) * 100)) + (((int) coreId) * 25);
                        int dst_offset = 0;
                        for (int i = 0; i < 25; ++i)
                        {
                            input0_local_nram[dst_offset + i] = input0[src_offset + i];
                        }
                    }
                    if ((((i0_outer_outer_outer * 16) + (((int) clusterId) * 4)) + ((int) coreId)) < 45)
                    {
                        // Detensorizing the __bang_active_sign
                        for (int i = 0; i < 25; ++i)
                        {
                            if (input0_local_nram[i] >= 0)
                                input0_local_nram[i] = 1.0f;
                            else
                                input0_local_nram[i] = -1.0f;
                        }
                    }
                    if ((((i0_outer_outer_outer * 16) + (((int) clusterId) * 4)) + ((int) coreId)) < 45)
                    {
                        int src_offset = 0;
                        int dst_offset = ((i0_outer_outer_outer * 400) + (((int) clusterId) * 100)) + (((int) coreId) * 25);
                        for (int i = 0; i < 25; ++i)
                        {
                            active_sign_147[dst_offset + i] = input0_local_nram[src_offset + i];
                        }
                    }
                }
            }
        }
    }
    """
    code = simplify_code(c_code)
    print(code)
