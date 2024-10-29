from pycparser import c_ast, c_generator, c_parser

from smt.util import NodeTransformer


class MergeForLoopsVisitor(NodeTransformer):
    def visit_Compound(self, node):
        """查找并合并连续的 for 循环"""
        if not node.block_items:
            return node
        new_block_items = []
        i = 0
        while i < len(node.block_items):
            current_node = node.block_items[i]
            # 检查是否是 for 循环，且下一个节点也是 for 循环
            if (
                isinstance(current_node, c_ast.For)
                and i + 1 < len(node.block_items)
                and isinstance(node.block_items[i + 1], c_ast.For)
            ):
                combined_body = []

                # 提取第一个循环的条件和步进
                init_stmt = current_node.init
                cond_stmt = current_node.cond
                next_stmt = current_node.next

                # 遍历连续的相似的 for 循环
                while (
                    i < len(node.block_items)
                    and isinstance(node.block_items[i], c_ast.For)
                    and self.is_similar_loop(current_node, node.block_items[i])
                ):
                    # 合并循环体内容
                    combined_body.extend(node.block_items[i].stmt.block_items)
                    i += 1  # 继续检查下一个节点

                # 创建合并后的 for 循环
                combined_for = c_ast.For(
                    init=init_stmt,
                    cond=cond_stmt,
                    next=next_stmt,
                    stmt=c_ast.Compound(block_items=combined_body),
                )

                # 将合并的循环添加到新的 block_items
                new_block_items.append(combined_for)
            else:
                # 非 for 循环节点直接添加
                new_block_items.append(current_node)
                i += 1

        # 更新 node 的 block_items
        node.block_items = new_block_items
        return node

    def is_similar_loop(self, loop1, loop2):
        """检查两个 for 循环是否具有相同的循环条件、初始条件和步进操作"""
        return (
            isinstance(loop1, c_ast.For)
            and isinstance(loop2, c_ast.For)
            and loop1.init.__class__ == loop2.init.__class__
            and loop1.cond.__class__ == loop2.cond.__class__
            and loop1.next.__class__ == loop2.next.__class__
            and loop1.init.decls[0].name == loop2.init.decls[0].name  # 确保变量名一致
            and loop1.cond.right.value == loop2.cond.right.value  # 确保范围一致
            and loop1.cond.op == loop2.cond.op  # 确保操作符一致
        )


# 示例使用
code = """
void add_kernel0(float *lhs, float *rhs, float *add_1515)
{
  float lhs_local_nram[128];
for (int i = 0; i < 64; i++)
{
    (((float *) lhs_local_nram) + 0)[i] = (((float *) lhs) + ((((int) clusterId) * 256) + (((int) coreId) * 64)))[i];
}

for (int i = 0; i < 64; i++)
{
    (((float *) lhs_local_nram) + 64)[i] = (((float *) rhs) + ((((int) clusterId) * 256) + (((int) coreId) * 64)))[i];
}

for (int i = 0; i < 64; i++)
{
    (((float *) lhs_local_nram) + 0)[i] = (((float *) lhs_local_nram) + 0)[i] + (((float *) lhs_local_nram) + 64)[i];
}


for (int i = 0; i < 64; i++)
{
    (((float *) add_1515) + ((((int) clusterId) * 256) + (((int) coreId) * 64)))[i] = (((float *) lhs_local_nram) + 0)[i];
}

}
"""

# 解析代码并应用合并
parser = c_parser.CParser()
ast = parser.parse(code)

# 使用 MergeForLoopsVisitor 进行遍历和合并
merge_visitor = MergeForLoopsVisitor()
ast = merge_visitor.visit(ast)

# 输出修改后的代码
generator = c_generator.CGenerator()
print(generator.visit(ast))
