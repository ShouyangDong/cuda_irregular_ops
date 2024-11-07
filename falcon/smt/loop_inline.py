from pycparser import c_ast, c_generator, c_parser

from falcon.smt.util import NodeTransformer


class InlineTransformer(NodeTransformer):
    def visit_Compound(self, node):
        """递归地遍历 Compound 节点，内联所有可以内联的语句"""
        node.block_items = self.inline_statements(node.block_items)
        return self.generic_visit(node)

    def inline_statements(self, block_items):
        """递归地检查和内联 block_items 中的语句"""
        if not block_items:
            return []

        new_block_items = []
        i = 0
        while i < len(block_items):
            stmt = block_items[i]

            if i > 0 and isinstance(stmt, c_ast.Assignment):
                # 获取前一个语句
                prev_stmt = new_block_items[-1]

                # 检查是否可以内联
                if isinstance(prev_stmt, c_ast.Assignment) and self.is_dependency(
                    prev_stmt, stmt
                ):
                    # 内联当前语句到前一个语句
                    inlined_stmt = self.create_inlined_stmt(prev_stmt, stmt)
                    new_block_items[-1] = inlined_stmt  # 更新最后一个语句为内联后的语句

                    # 更新后继续递归内联
                    new_block_items = self.inline_statements(new_block_items)
                    i += 1
                    continue  # 跳过当前语句，已经被内联
                else:
                    new_block_items.append(stmt)
            else:
                new_block_items.append(stmt)

            i += 1

        return new_block_items

    def is_dependency(self, stmt1, stmt2):
        """检查 stmt1 的输出是否是 stmt2 的输入"""
        if isinstance(stmt2.rvalue, c_ast.FuncCall):
            return self.nodes_equal(stmt1.lvalue, stmt2.rvalue.args.exprs[0])

        elif isinstance(stmt2.rvalue, c_ast.UnaryOp):
            return self.nodes_equal(stmt1.lvalue, stmt2.rvalue.expr)

        elif isinstance(stmt2.rvalue, c_ast.BinaryOp):
            return self.nodes_equal(
                stmt1.lvalue, stmt2.rvalue.left
            ) or self.nodes_equal(stmt1.lvalue, stmt2.rvalue.right)

        else:
            return self.nodes_equal(stmt1.lvalue, stmt2.rvalue)

    def create_inlined_stmt(self, stmt1, stmt2):
        """创建一个新的内联语句，将 stmt2 的右值更新为 stmt1 的左值"""
        if isinstance(stmt2.rvalue, c_ast.FuncCall):
            inlined_stmt = c_ast.Assignment(
                op="=",
                lvalue=stmt2.lvalue,
                rvalue=c_ast.FuncCall(
                    name=stmt2.rvalue.name, args=c_ast.ExprList([stmt1.rvalue])
                ),
            )
        elif isinstance(stmt2.rvalue, c_ast.UnaryOp):
            inlined_stmt = c_ast.Assignment(
                op="=",
                lvalue=stmt2.lvalue,
                rvalue=c_ast.UnaryOp(op=stmt2.rvalue.op, expr=stmt1.rvalue),
            )
        elif isinstance(stmt2.rvalue, c_ast.BinaryOp):
            if self.nodes_equal(stmt1.lvalue, stmt2.rvalue.left):
                inlined_stmt = c_ast.Assignment(
                    op="=",
                    lvalue=stmt2.lvalue,
                    rvalue=c_ast.BinaryOp(
                        op=stmt2.rvalue.op, left=stmt1.rvalue, right=stmt2.rvalue.right
                    ),
                )
            else:
                inlined_stmt = c_ast.Assignment(
                    op="=",
                    lvalue=stmt2.lvalue,
                    rvalue=c_ast.BinaryOp(
                        op=stmt2.rvalue.op, left=stmt2.rvalue.left, right=stmt1.rvalue
                    ),
                )
        else:
            inlined_stmt = c_ast.Assignment(
                op="=", lvalue=stmt2.lvalue, rvalue=stmt1.rvalue
            )
        return inlined_stmt

    def nodes_equal(self, node1, node2):
        """递归地比较两个 AST 节点是否相同"""
        generator = c_generator.CGenerator()
        output_code = generator.visit(node1)
        generator = c_generator.CGenerator()
        output_code = generator.visit(node2)
        return output_code == output_code


def ast_inline(code):
    # 解析代码
    parser = c_parser.CParser()
    ast = parser.parse(code)

    # 进行内联转换
    transformer = InlineTransformer()
    ast = transformer.visit(ast)

    # 输出修改后的代码
    generator = c_generator.CGenerator()
    return generator.visit(ast)


if __name__ == "__main__":

    # 输入代码
    code = """
    void tanh(float *input0, float *active_tanh_210)
    {
    for (int clusterId = 0; clusterId < 4; ++clusterId)
    {
        for (int coreId = 0; coreId < 4; ++coreId)
        {
        float input0_local_nram[640];
        for (int i = 0; i < 640; i++)
        {
            (((float *) input0_local_nram) + 0)[i] = (((float *) input0) + ((((int) clusterId) * 2560) + (((int) coreId) * 640)))[i];
            (((float *) input0_local_nram) + 0)[i] = tanh((((float *) input0_local_nram) + 0)[i]);
            (((float *) active_tanh_210) + ((((int) clusterId) * 2560) + (((int) coreId) * 640)))[i] = (((float *) input0_local_nram) + 0)[i];
        }
        }
    }
    }
    """
    code = ast_inline(code)
    print(code)

    # 输入代码
    code = """
    void tanh(float *input0, float *active_tanh_210)
    {
    for (int clusterId = 0; clusterId < 4; ++clusterId)
    {
        for (int coreId = 0; coreId < 4; ++coreId)
        {
        float input0_local_nram[640];
        for (int i = 0; i < 640; i++)
        {
            (((float *) input0_local_nram) + 0)[i] = (((float *) input0) + ((((int) clusterId) * 2560) + (((int) coreId) * 640)))[i];
            (((float *) input0_local_nram) + 0)[i] = -(((float *) input0_local_nram) + 0)[i];
            (((float *) active_tanh_210) + ((((int) clusterId) * 2560) + (((int) coreId) * 640)))[i] = (((float *) input0_local_nram) + 0)[i];
        }
        }
    }
    }
    """
    code = ast_inline(code)
    print(code)
