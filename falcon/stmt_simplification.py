import re

from pycparser import c_ast, c_generator, c_parser

from falcon.smt.util import NodeTransformer


class MergeLoopsAndIfsVisitor(NodeTransformer):
    def __init__(self):
        self.loop_vars = []
        self.loop_bounds = {}

    def visit_For(self, node):
        if isinstance(node.cond.right, c_ast.Constant) and isinstance(
            node.init, c_ast.DeclList
        ):
            if node.cond.op == "<":
                self.loop_bounds[node.init.decls[0].name] = (
                    int(node.cond.right.value) - 1
                )
            elif node.cond.op == "<=":
                self.loop_bounds[node.init.decls[0].name] = int(node.cond.right.value)
        return self.generic_visit(node)

    def visit_Compound(self, node):
        """查找并合并连续的 for 循环和 if 语句"""
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
                    and self.nodes_equal(current_node, node.block_items[i])
                ):
                    loop_var = self.get_loop_variable(node.block_items[i])
                    if loop_var:
                        self.loop_vars.append(loop_var)

                    # 合并循环体内容
                    combined_body.extend(node.block_items[i].stmt.block_items)
                    i += 1  # 继续检查下一个节点

                # 更新循环体中的循环变量
                combined_body = self.rename_loop_variables(combined_body)

                # 创建合并后的 for 循环
                combined_for = c_ast.For(
                    init=init_stmt,
                    cond=cond_stmt,
                    next=next_stmt,
                    stmt=c_ast.Compound(block_items=combined_body),
                )

                # 将合并的循环添加到新的 block_items
                new_block_items.append(combined_for)
            # 检查是否是 if 语句，且下一个节点也是 if 语句
            elif (
                isinstance(current_node, c_ast.If)
                and i + 1 < len(node.block_items)
                and isinstance(node.block_items[i + 1], c_ast.If)
                and self.nodes_equal(current_node, node.block_items[i + 1])
            ):
                combined_body = []

                # 遍历连续的相似的 if 语句
                while (
                    i < len(node.block_items)
                    and isinstance(node.block_items[i], c_ast.If)
                    and self.nodes_equal(current_node, node.block_items[i])
                ):
                    # 合并 if 语句的主体内容
                    if node.block_items[i].iftrue:
                        combined_body.extend(node.block_items[i].iftrue.block_items)
                    i += 1  # 继续检查下一个节点

                # 创建合并后的 if 语句
                combined_if = c_ast.If(
                    cond=current_node.cond,
                    iftrue=c_ast.Compound(block_items=combined_body),
                    iffalse=None,
                )

                # 将合并的 if 语句添加到新的 block_items
                new_block_items.append(combined_if)
            else:
                # 非 for 循环或 if 语句节点直接添加
                new_block_items.append(current_node)
                i += 1

        # 更新 node 的 block_items
        node.block_items = new_block_items
        return self.generic_visit(node)

    def is_similar_loop(self, loop1, loop2):
        """检查两个 for 循环是否具有相同的循环条件、初始条件和步进操作"""
        return (
            isinstance(loop1, c_ast.For)
            and isinstance(loop2, c_ast.For)
            and loop1.init.__class__ == loop2.init.__class__
            and loop1.cond.__class__ == loop2.cond.__class__
            and loop1.next.__class__ == loop2.next.__class__
            and loop1.cond.right.value == loop2.cond.right.value  # 确保范围一致
            and loop1.cond.op == loop2.cond.op  # 确保操作符一致
        )

    def get_loop_variable(self, for_loop):
        """获取 for 循环中的循环变量"""
        if isinstance(for_loop, c_ast.For):
            return for_loop.init.decls[0].name
        return None

    def rename_loop_variables(self, block_items):
        """重命名循环体中的循环变量"""
        # 使用第一个循环变量的名称进行重命名
        for item in block_items:
            self.generic_visit(item)
        return block_items

    def visit_ID(self, node):
        if node.name in self.loop_vars:
            node.name = self.loop_vars[0]
        return node

    def nodes_equal(self, node1, node2):
        """递归地比较两个 AST 节点是否相同"""
        generator = c_generator.CGenerator()
        output_code = generator.visit(node1)
        generator = c_generator.CGenerator()
        output_code = generator.visit(node2)
        return output_code == output_code

    def visit_If(self, node):
        # 检查 if 条件是否恒为真
        if self.is_condition_always_true(node.cond):
            # 如果条件恒为真，将 if 语句替换为其主体内容
            return node.iftrue.block_items
        # 否则，保留 if 语句
        return self.generic_visit(node)

    def is_condition_always_true(self, condition):
        # 只处理简单的 `<` 和 `<=` 情况
        if isinstance(condition, c_ast.BinaryOp) and condition.op in ("<", "<="):
            left, right = condition.left, condition.right
            # 判断右侧是否为一个常量
            if isinstance(right, c_ast.Constant):
                # 检查左侧是否包含循环变量及范围，并确定条件恒真
                return self.is_left_expression_in_bounds(left, int(right.value))
        return False

    def is_left_expression_in_bounds(self, expr, bound):
        # 递归检查表达式的最大可能值是否小于给定的界限
        expr_bound = self.get_expression_bound(expr)
        return expr_bound is not None and expr_bound < bound

    def get_expression_bound(self, expr):
        # 计算表达式的最大可能值，用于判断是否在上界之内
        if isinstance(expr, c_ast.ID):
            # 获取变量的范围上界
            return self.get_variable_bound(expr.name)
        elif isinstance(expr, c_ast.Constant):
            # 如果是常量，返回其值
            return int(expr.value)
        elif isinstance(expr, c_ast.BinaryOp):
            left_bound = self.get_expression_bound(expr.left)
            right_bound = self.get_expression_bound(expr.right)
            if expr.op == "+":
                # 处理加法操作
                if left_bound is not None and right_bound is not None:
                    return left_bound + right_bound
            elif expr.op == "*":
                # 处理乘法操作
                if left_bound is not None and right_bound is not None:
                    return left_bound * right_bound
            elif expr.op == "-":
                # 处理减法操作
                if left_bound is not None and right_bound is not None:
                    return left_bound - right_bound
            elif expr.op == "/":
                # 处理除法操作，避免除以零
                if (
                    left_bound is not None
                    and right_bound is not None
                    and right_bound != 0
                ):
                    return left_bound // right_bound  # 整数除法
        return None

    def get_variable_bound(self, var_name):
        # 返回循环变量的上界（假设这些变量的范围是已知的）
        return self.loop_bounds.get(var_name, None)


def ast_stmt_simplification(code):
    code = re.sub(r"//.*?\n|/\*.*?\*/", "", code, flags=re.S)
    # 解析代码并应用合并
    parser = c_parser.CParser()
    ast = parser.parse(code)
    # 使用 MergeForLoopsVisitor 进行遍历和合并
    merge_visitor = MergeLoopsAndIfsVisitor()
    ast = merge_visitor.visit(ast)

    # 输出修改后的代码
    generator = c_generator.CGenerator()
    return generator.visit(ast)


if __name__ == "__main__":
    # 示例使用
    code = """
    void add(float *lhs, float *rhs, float *add_1515)
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
    code = ast_stmt_simplification(code)
    print(code)

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
        }

        for (int i = 0; i < 640; i++)
        {
            (((float *) input0_local_nram) + 0)[i] = tanh((((float *) input0_local_nram) + 0)[i]);
        }

        for (int i = 0; i < 640; i++)
        {
            (((float *) active_tanh_210) + ((((int) clusterId) * 2560) + (((int) coreId) * 640)))[i] = (((float *) input0_local_nram) + 0)[i];
        }

        }

    }

    }
    """
    code = ast_stmt_simplification(code)
    print(code)

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
        }

        for (int j = 0; j < 640; j++)
        {
            (((float *) input0_local_nram) + 0)[j] = tanh((((float *) input0_local_nram) + 0)[j]);
        }

        for (int k = 0; k < 640; k++)
        {
            (((float *) active_tanh_210) + ((((int) clusterId) * 2560) + (((int) coreId) * 640)))[k] = (((float *) input0_local_nram) + 0)[k];
        }

        }

    }

    }
    """
    code = ast_stmt_simplification(code)
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
    code = ast_stmt_simplification(c_code)
    print(code)

    c_code = """
    void add(float *A, float *B, float *T_add)
    {
        for (int k = 0; k < 16; k++)
        {
            for (int j = 0; j < 256; j++)
            {
                if (((k * 256) + j) < 4096)
                {
                    T_add[(k * 256) + j] = A[(k * 256) + j] + B[(k * 256) + j];
                }
            }

        }
    }
    """
    code = ast_stmt_simplification(c_code)
    print(code)
