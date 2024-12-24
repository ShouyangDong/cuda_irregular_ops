from pycparser import c_ast, c_generator, c_parser

from falcon.util import NodeTransformer


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
                if isinstance(
                    prev_stmt, c_ast.Assignment
                ) and self.is_dependency(prev_stmt, stmt):
                    # 内联当前语句到前一个语句
                    inlined_stmt = self.create_inlined_stmt(prev_stmt, stmt)
                    new_block_items[-1] = (
                        inlined_stmt  # 更新最后一个语句为内联后的语句
                    )

                    # 继续递归内联
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
        # 确认 stmt1 的左值是否被 stmt2 右值引用
        return self.nodes_equal(
            stmt1.lvalue, stmt2.rvalue
        ) or self.contains_reference(stmt2.rvalue, stmt1.lvalue)

    def create_inlined_stmt(self, stmt1, stmt2):
        """创建一个新的内联语句，将 stmt2 的右值更新为 stmt1 的左值"""
        # 创建一个新的内联赋值语句
        inlined_stmt = c_ast.Assignment(
            op="=", lvalue=stmt2.lvalue, rvalue=stmt2.rvalue
        )

        # 将 stmt2 的右值中的引用替换为 stmt1 的右值
        inlined_stmt.rvalue = self.replace_node(
            stmt2.rvalue, stmt1.lvalue, stmt1.rvalue
        )
        return inlined_stmt

    def replace_node(self, node, target, replacement):
        """将 AST 中的 target 节点替换为 replacement 节点"""
        if self.nodes_equal(node, target):
            return replacement
        for child_name, child in node.children():
            setattr(
                node, child_name, self.replace_node(child, target, replacement)
            )
        return node

    def contains_reference(self, node, target):
        """检查节点是否包含对目标的引用"""
        if self.nodes_equal(node, target):
            return True
        for _, child in node.children():
            if self.contains_reference(child, target):
                return True
        return False

    def nodes_equal(self, node1, node2):
        """递归地比较两个 AST 节点是否相同"""
        generator = c_generator.CGenerator()
        return generator.visit(node1) == generator.visit(node2)


class UnusedMemoryRemover(NodeTransformer):
    def visit_Compound(self, node):
        """遍历 Compound 节点，删除未使用的内存分配语句"""
        node.block_items = self.remove_unused_allocations(node.block_items)
        return self.generic_visit(node)

    def remove_unused_allocations(self, block_items):
        """检查并删除未使用的内存分配语句"""
        new_block_items = []
        for i in range(len(block_items)):
            stmt = block_items[i]
            # 如果是内存分配语句，并且其变量未被后续语句引用，则删除该语句
            if isinstance(stmt, c_ast.Decl) and isinstance(
                stmt.type, c_ast.ArrayDecl
            ):
                if not any(
                    self.contains_reference(later_stmt, stmt.name)
                    for later_stmt in block_items[i + 1 :]
                ):
                    continue  # 跳过未使用的内存分配语句
            if isinstance(stmt, c_ast.Decl) and isinstance(
                stmt.type, c_ast.TypeDecl
            ):
                if isinstance(stmt.type.type, c_ast.ArrayDecl):
                    if not any(
                        self.contains_reference(later_stmt, stmt.name)
                        for later_stmt in block_items[i + 1 :]
                    ):
                        continue  # 删除未使用的数组声明
            new_block_items.append(stmt)
        return new_block_items

    def contains_reference(self, node, target):
        """检查节点是否包含对目标的引用"""
        if hasattr(node, "name") and node.name == target:
            return True
        for _, child in node.children():
            if self.contains_reference(child, target):
                return True
        return False


def ast_buffer_inline(code):
    # 解析代码
    parser = c_parser.CParser()
    ast = parser.parse(code)

    # 进行内联转换
    transformer = InlineTransformer()
    ast = transformer.visit(ast)

    # 删除未使用的内存分配
    remover = UnusedMemoryRemover()
    ast = remover.visit(ast)
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
    code = ast_buffer_inline(code)
    print(code)
