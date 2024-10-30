from pycparser import c_ast, c_generator, c_parser


class SplitForLoopVisitor(c_ast.NodeVisitor):
    def __init__(self):
        self.factor = None  # 用于存储从pragma中提取的拆分因子
        self.axis_name = None
        self.org_extent = None

    def visit_Compound(self, node):
        """查找 #pragma loop_split 并获取拆分因子，应用到后续的 for 循环"""
        blocks = node.block_items
        if not blocks:
            return

        new_block_items = []
        skip_next = False

        # 遍历 `block_items`
        for index, subnode in enumerate(blocks):
            if skip_next:
                skip_next = False
                continue

            # 检查是否是 `#pragma loop_split(<factor>)`
            if isinstance(subnode, c_ast.Pragma) and "loop_split" in subnode.string:
                # 提取因子值
                pragma_content = subnode.string.strip()
                self.factor = int(pragma_content.split("(")[-1].split(")")[0])

                # 检查下一节点是否为 `for` 循环
                if index + 1 < len(blocks) and isinstance(blocks[index + 1], c_ast.For):
                    self.axis_name = blocks[index + 1].init.decls[0].name
                    # 应用循环拆分
                    split_for_loop = self.split_for_loop(blocks[index + 1])
                    new_block_items.append(split_for_loop)

                    # 跳过下一节点的 `for` 循环
                    skip_next = True
                else:
                    # 不是 `for` 循环的情况，添加原节点
                    new_block_items.append(subnode)
            else:
                # 如果不是 `#pragma loop_split` 或者 `for`，直接添加节点
                new_block_items.append(subnode)

        # 更新 `block_items`
        node.block_items = new_block_items
        self.generic_visit(node)

    def split_for_loop(self, node):
        """对 for 循环进行拆分"""
        # 提取原始循环的最大值（循环范围）
        self.org_extent = int(node.cond.right.value)
        outer_extent = self.factor

        # 创建内部循环
        inner_init = c_ast.Decl(
            name=self.axis_name + "_in",
            quals=[],
            align=[],
            storage=[],
            funcspec=[],
            type=c_ast.TypeDecl(
                declname=self.axis_name + "_in",
                quals=[],
                align=None,
                type=c_ast.IdentifierType(["int"]),
            ),
            init=c_ast.Constant("int", "0"),
            bitsize=None,
        )
        inner_cond = c_ast.BinaryOp(
            node.cond.op,
            c_ast.ID(self.axis_name + "_in"),
            c_ast.Constant("int", str(self.org_extent // self.factor)),
        )
        inner_next = c_ast.UnaryOp(node.next.op, c_ast.ID(self.axis_name + "_in"))

        # 内层循环的 `for` 结构
        inner_for = c_ast.For(
            init=inner_init,
            cond=inner_cond,
            next=inner_next,
            stmt=node.stmt,
        )

        # 将内层循环包裹在一个 `Compound` 块中
        inner_compound = c_ast.Compound(block_items=[inner_for])

        # 外层循环
        outer_init = c_ast.Decl(
            name=self.axis_name + "_out",
            quals=[],
            align=[],
            storage=[],
            funcspec=[],
            type=c_ast.TypeDecl(
                declname=self.axis_name + "_out",
                quals=[],
                align=None,
                type=c_ast.IdentifierType(["int"]),
            ),
            init=c_ast.Constant("int", "0"),
            bitsize=None,
        )
        outer_cond = c_ast.BinaryOp(
            node.cond.op,
            c_ast.ID(self.axis_name + "_out"),
            c_ast.Constant("int", str(outer_extent)),
        )
        outer_next = c_ast.UnaryOp(node.next.op, c_ast.ID(self.axis_name + "_out"))

        # 外层循环的 `for` 结构
        outer_for = c_ast.For(
            init=outer_init,
            cond=outer_cond,
            next=outer_next,
            stmt=inner_compound,
        )

        # 修改内层循环中对 `axis_name` 的引用
        self.generic_visit(inner_for)
        return outer_for

    def visit_ID(self, node):
        if node.name == self.axis_name:
            node.name = (
                self.axis_name
                + "_out"
                + " * "
                + str(self.org_extent // self.factor)
                + " + "
                + self.axis_name
                + "_in"
            )


def ast_loop_split(code):
    # Parse the C code
    parser = c_parser.CParser()
    ast = parser.parse(code)
    generator = c_generator.CGenerator()
    # Custom visitor instance
    visitor = SplitForLoopVisitor()
    # Visit the AST to split 'for' loops with loop count 10 into 2 loops with counts 2 and 5
    visitor.visit(ast)
    return generator.visit(ast)


if __name__ == "__main__":
    code = """
    int factorial(int result) {
        #pragma loop_split(2)
        for (int i = 0; i < 10; i++) {
            result += i;
        }
        return result;
    }
    """
    final_node = ast_loop_split(code)
    print(final_node)

    code = """
    void add_kernel(float* A, float* B, float* T_add) {
        for (int i = 0; i < 256; i++) {
            #pragma loop_split(4)
            for(int j = 0; j < 1024; j++) {
                T_add[((i * 1024) + j)] = (A[((i * 1024) + j)] + B[((i * 1024) + j)]);
            }
        }
    }
    """
    final_node = ast_loop_split(code)
    print(final_node)
