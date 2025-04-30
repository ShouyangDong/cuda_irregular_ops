from pycparser import c_ast

from falcon.simplification import simplify_code
from falcon.util import NodeTransformer, generate_code, parse_code_ast


class LoopNestFusionVisitor(NodeTransformer):
    def __init__(self):
        self.outer_var = None
        self.inner_var = None
        self.inner_bound = None

    def visit_For(self, node):
        # 检查当前 for 循环是否包含嵌套的 for 循环
        if (
            isinstance(node.stmt, c_ast.Compound)
            and len(node.stmt.block_items) == 1
        ):
            nested_loop = node.stmt.block_items[0]
            if isinstance(nested_loop, c_ast.For) and isinstance(
                node.next, c_ast.UnaryOp
            ):
                # 合并嵌套的 for 循环
                fused_loop = self.fuse_loops(node, nested_loop)
                node.init = fused_loop.init
                node.cond = fused_loop.cond
                node.next = fused_loop.next
                node.stmt = fused_loop.stmt

                self.outer_var = node.init.decls[0].name
                self.inner_var = nested_loop.init.decls[0].name
                self.inner_bound = nested_loop.cond.right.value

                # 更新初始化部分，将变量名改为 i_j_fused
                fused_name = f"{node.init.decls[0].name}_{nested_loop.init.decls[0].name}_fused"
                node.init.decls[0].name = fused_name
                node.init.decls[0].type.declname = fused_name
                node.cond.left.name = fused_name
                node.next.expr.name = fused_name
                return self.generic_visit(node)

        return self.generic_visit(node)

    def visit_BinaryOp(self, node):
        if node.op == "+":
            left = node.left
            right = node.right
            if (
                isinstance(right, c_ast.ID)
                and right.name == self.inner_var
                and isinstance(left, c_ast.BinaryOp)
                and left.op == "*"
                and isinstance(left.right, c_ast.Constant)
                and left.right.value == self.inner_bound
                and isinstance(left.left, c_ast.ID)
                and left.left.name == self.outer_var
            ):
                # 替换为融合变量
                fused_index = c_ast.ID(
                    name=f"{self.outer_var}_{self.inner_var}_fused"
                )
                return fused_index
        return self.generic_visit(node)

    def visit_ArrayRef(self, node):
        if (
            isinstance(node.name, c_ast.BinaryOp)
            and node.subscript.name == self.inner_var
        ):
            right = node.name.right
            if (
                isinstance(right, c_ast.BinaryOp)
                and right.left.expr.name == self.outer_var
                and right.op == "*"
                and isinstance(right.right, c_ast.Constant)
                and right.right.value == self.inner_bound
            ):
                fused_index = c_ast.ID(
                    name=f"{self.outer_var}_{self.inner_var}_fused"
                )
                node.subscript = fused_index
                node.name = node.name.left
            return node
        return self.generic_visit(node)

    def fuse_loops(self, outer_loop, inner_loop):
        # 创建新的循环变量和边界，合并两个循环
        fused_init = outer_loop.init
        fused_cond = c_ast.BinaryOp(
            op="<",
            left=c_ast.ID(name=fused_init.decls[0].name),
            right=c_ast.Constant(
                type="int",
                value=str(
                    int(outer_loop.cond.right.value)
                    * int(inner_loop.cond.right.value)
                ),
            ),
        )
        fused_next = outer_loop.next
        # 将嵌套循环的主体语句调整为单层循环
        fused_stmt = c_ast.Compound(inner_loop.stmt.block_items)
        fused_loop = c_ast.For(
            init=fused_init, cond=fused_cond, next=fused_next, stmt=fused_stmt
        )
        return fused_loop


def ast_loop_fusion(c_code):
    ast = parse_code_ast(c_code)
    # 自定义访问者实例
    visitor = LoopNestFusionVisitor()
    # 访问 AST，合并可以合并的嵌套 for 循环
    visitor.visit(ast)
    code = generate_code(ast)
    code = simplify_code(code)
    return code


if __name__ == "__main__":
    # 含有嵌套 for 循环的 C 代码
    c_code = """
    void multiply_matrices(int* a, int* b, int* result) {
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                result[i * 10 + j] = a[i * 10 + j] * b[i * 10 + j];
            }
        }
    }
    """
    # 合并循环后的最终代码
    final_code = ast_loop_fusion(c_code)
    print(final_code)

    c_code = """
    void add(float *lhs, float *rhs, float *add_1935)
    {
        for (int coreId = 0; coreId < 4; ++coreId)
        {
            for (int i = 0; i < 1024; i++)
            {
            (((float *) add_1935) + (((int) coreId) * 1024))[i] = (((float *) lhs) + (((int) coreId) * 1024))[i] + (((float *) rhs) + (((int) coreId) * 1024))[i];
            }

        }
    }
    """
    # 合并循环后的最终代码
    final_code = ast_loop_fusion(c_code)
    print(final_code)

    c_code = """
    void add(float *A, float *B, float *T_add)
    {
        for (int blockIdxx_threadIdxx_fused = 0; blockIdxx_threadIdxx_fused < 262144; ++blockIdxx_threadIdxx_fused)
        {
            if (blockIdxx_threadIdxx_fused < 4096)
            {
                T_add[(((int) blockIdxx) * 1024) + ((int) threadIdxx)] = A[(((int) blockIdxx) * 1024) + ((int) threadIdxx)] + B[(((int) blockIdxx) * 1024) + ((int) threadIdxx)];
            }
        }

    }
    """
    # 合并循环后的最终代码
    final_code = ast_loop_fusion(c_code)
    print(final_code)
