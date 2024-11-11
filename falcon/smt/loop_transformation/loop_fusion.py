from pycparser import c_ast, c_generator, c_parser


class LoopNestFusionVisitor(c_ast.NodeVisitor):
    def visit_For(self, node):
        # 检查当前 for 循环是否包含嵌套的 for 循环
        if isinstance(node.stmt, c_ast.Compound) and len(node.stmt.block_items) == 1:
            nested_loop = node.stmt.block_items[0]
            if isinstance(nested_loop, c_ast.For):
                # 合并嵌套的 for 循环
                fused_loop = self.fuse_loops(node, nested_loop)
                node.init = fused_loop.init
                node.cond = fused_loop.cond
                node.next = fused_loop.next
                node.stmt = fused_loop.stmt
                self.update_loop_index(
                    node.stmt,
                    outer_var=node.init.decls[0].name,
                    inner_var=nested_loop.init.decls[0].name,
                    inner_bound=nested_loop.cond.right.value,
                )
                # 更新初始化部分，将变量名改为 i_j_fused
                node.init.decls[0].name = (
                    node.init.decls[0].name
                    + "_"
                    + nested_loop.init.decls[0].name
                    + "_fused"
                )
                node.init.decls[0].type.declname = node.init.decls[0].name
                node.cond.left.name = node.init.decls[0].name
                node.next.expr.name = node.init.decls[0].name

        # 递归访问子节点
        self.generic_visit(node)

    def fuse_loops(self, outer_loop, inner_loop):
        # 创建新的循环变量和边界，合并两个循环
        fused_init = outer_loop.init
        fused_cond = c_ast.BinaryOp(
            op="<",
            left=c_ast.ID(name=fused_init.decls[0].name),
            right=c_ast.Constant(
                type="int",
                value=str(
                    int(outer_loop.cond.right.value) * int(inner_loop.cond.right.value)
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

    def update_loop_index(self, node, outer_var, inner_var, inner_bound):
        # 更新循环体中的索引表达式，将 i + (10 * j) 转换为单一索引 i_j_fused
        class IndexUpdater(c_ast.NodeVisitor):
            def visit_ArrayRef(self, array_node):
                if (
                    isinstance(array_node.subscript, c_ast.BinaryOp)
                    and array_node.subscript.op == "+"
                ):
                    left = array_node.subscript.left
                    right = array_node.subscript.right
                    if (
                        isinstance(right, c_ast.ID)
                        and right.name == inner_var
                        and isinstance(left, c_ast.BinaryOp)
                        and left.op == "*"
                        and isinstance(left.right, c_ast.Constant)
                        and left.right.value == str(inner_bound)
                        and isinstance(left.left, c_ast.ID)
                        and left.left.name == outer_var
                    ):
                        # 替换索引为单一变量 i_j_fused
                        fused_index = c_ast.ID(
                            name=outer_var + "_" + inner_var + "_fused"
                        )
                        array_node.subscript = fused_index

                elif (
                    isinstance(array_node.name, c_ast.BinaryOp)
                    and array_node.subscript.name == inner_var
                ):
                    right = array_node.name.right
                    if (
                        isinstance(right, c_ast.BinaryOp)
                        and right.left.expr.name == outer_var
                        and right.op == "*"
                        and isinstance(right.right, c_ast.Constant)
                        and right.right.value == str(inner_bound)
                    ):
                        fused_index = c_ast.ID(
                            name=outer_var + "_" + inner_var + "_fused"
                        )
                        array_node.subscript = fused_index
                        array_node.name = array_node.name.left

                self.generic_visit(array_node)

        updater = IndexUpdater()
        updater.visit(node)


def ast_loop_fusion(c_code):
    # 解析 C 代码
    parser = c_parser.CParser()
    ast = parser.parse(c_code)
    generator = c_generator.CGenerator()
    # 自定义访问者实例
    visitor = LoopNestFusionVisitor()
    # 访问 AST，合并可以合并的嵌套 for 循环
    visitor.visit(ast)
    return generator.visit(ast)


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
