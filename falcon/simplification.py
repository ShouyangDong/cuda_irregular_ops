import re

from pycparser import c_ast, c_generator, c_parser

from falcon.smt.util import NodeTransformer


class SimplifyConstants(NodeTransformer):
    def __init__(self):
        self.for_loop_map = {}

    def visit_For(self, node):
        self.for_loop_map[node.init.decls[0].name] = node.cond.right.value
        return self.generic_visit(node)

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
            return self.generic_visit(node)
        elif node.op == "+":
            # 如果两个操作数都是常量，则可以进行简化
            if isinstance(node.left, c_ast.Constant) and isinstance(
                node.right, c_ast.Constant
            ):
                # 计算并返回新的常量节点
                result = int(node.left.value) + int(node.right.value)
                return c_ast.Constant("int", value=str(result))
            return self.generic_visit(node)
        if node.op == "/":
            # 如果两个操作数都是常量，则可以进行简化
            if isinstance(node.left, c_ast.Constant) and isinstance(
                node.right, c_ast.Constant
            ):
                # 计算并返回新的常量节点
                result = int(node.left.value) // int(node.right.value)
                return c_ast.Constant("int", value=str(result))
            return self.generic_visit(node)
        elif node.op == "-":
            # 如果两个操作数都是常量，则可以进行简化
            if isinstance(node.left, c_ast.Constant) and isinstance(
                node.right, c_ast.Constant
            ):
                # 计算并返回新的常量节点
                result = int(node.left.value) - int(node.right.value)
                return c_ast.Constant("int", value=str(result))
            return self.generic_visit(node)
        else:
            return self.generic_visit(node)

    def visit_If(self, node):
        if (
            isinstance(node.cond, c_ast.BinaryOp)
            and node.cond.op == "<"
            and isinstance(node.cond.left, c_ast.ID)
            and node.cond.left.name in self.for_loop_map
            and self.for_loop_map[node.cond.left.name] == node.cond.right.value
        ):
            # TODO:this may be a special case
            #  or (
            #     isinstance(node.cond, c_ast.BinaryOp)
            #     and node.cond.op == "<"
            #     and node.cond.right.value == "4"
            #     and (node.cond.left.name == "coreId" or node.cond.left.name == "clusterId")
            # ):
            return node.iftrue.block_items
        return self.generic_visit(node)

    def visit_For(self, node):
        if isinstance(node.stmt, c_ast.Compound):
            if len(node.stmt.block_items) == 1:
                if isinstance(node.stmt.block_items[0], c_ast.If):
                    stmt = node.stmt.block_items[0]
                    if stmt.iffalse is None:
                        if (
                            isinstance(stmt.cond, c_ast.BinaryOp)
                            and stmt.cond.op == "<"
                        ):
                            # 获取上限值
                            if_upper_bound = stmt.cond.right.value

                            if (
                                isinstance(node.cond, c_ast.BinaryOp)
                                and node.cond.op == "<"
                            ):
                                if int(node.cond.right.value) >= int(if_upper_bound):
                                    # 用找到的 `if` 语句中的值替换 `for` 循环中的上限值
                                    node.cond.right.value = if_upper_bound
                                    node.stmt = stmt.iftrue
                                    return self.generic_visit(node)
        return self.generic_visit(node)

    def visit_Cast(self, node):
        if node.to_type.type.type.names[0] == "int" and isinstance(node.expr, c_ast.ID):
            return node.expr
        return self.generic_visit(node)


def simplify_code(source_code):
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

    c_code = """
    void add(float *A, float *B, float *T_add)
    {
    for (int blockIdxx_threadIdxx_fused = 0; blockIdxx_threadIdxx_fused < 262144; ++blockIdxx_threadIdxx_fused)
    {
        if (blockIdxx_threadIdxx_fused < 4096)
        {
        T_add[blockIdxx_threadIdxx_fused] = A[blockIdxx_threadIdxx_fused] + B[blockIdxx_threadIdxx_fused];
        }
    }
    }
    """
    code = simplify_code(c_code)
    print(code)

    c_code = """
    void add(float *A, float *B, float *T_add)
    {
    for (int blockIdxx = 0; blockIdxx < 256; ++blockIdxx)
    {
        for (int threadIdxx = 0; threadIdxx < 1024; ++threadIdxx)
        {
        if (((blockIdxx * 1024) + threadIdxx) < 4096)
        {
            T_add[(((int) blockIdxx) * 1024) + ((int) threadIdxx)] = A[(((int) blockIdxx) * 1024) + ((int) threadIdxx)] + B[(((int) blockIdxx) * 1024) + ((int) threadIdxx)];
        }
        }

    }
    }
    """
    code = simplify_code(c_code)
    print(code)

    cuda_code = """
    softmax(float* A, float* T_softmax_norm) {
        if (threadIdx.x < 5) {
            int rowStart = threadIdx.x * 128;

            float maxVal = A[rowStart];
            for (int i = 1; i < 128; ++i) {
                if (A[rowStart + i] > maxVal) {
                    maxVal = A[rowStart + i];
                }
            }

            float denom = 0.0f;
            for (int i = 0; i < 128; ++i) {
                T_softmax_norm[rowStart + i] = expf(A[rowStart + i] - maxVal);
                denom += T_softmax_norm[rowStart + i];
            }

            for (int i = 0; i < 128; ++i) {
                T_softmax_norm[rowStart + i] /= denom;
            }
        }
    }
    """
    converted_code = simplify_code(cuda_code)
    print(converted_code)
