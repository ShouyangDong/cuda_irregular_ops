from pycparser import c_ast, c_generator, c_parser


class NodeTransformer(c_ast.NodeVisitor):
    """
    A node transformer that visits each node in an AST and applies transformations.

    Attributes:
    - None explicitly defined here, but subclasses may add attributes.
    """

    def generic_visit(self, node):
        """
        A generic visit method that is called for nodes that don't have a specific visit_<nodetype> method.

        This method iterates over all fields in the current node. If a field contains a list of nodes,
        it applies the transformation to each item in the list. If a field contains a single node, it applies
        the transformation to that node.

        Parameters:
        - node: The AST node to visit and potentially transform.

        Returns:
        - The original node, potentially with some of its fields transformed or replaced.
        """
        for field, old_value in iter_fields(node):
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, c_ast.Node):
                        value = self.visit(value)
                        if value is None:
                            continue
                        elif not isinstance(value, c_ast.Node):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, c_ast.Node):
                new_node = self.visit(old_value)
                setattr(node, field, new_node)
        return node


def iter_fields(node):
    """
    Iterate over all fields of a pycparser AST node.

    Parameters:
    - node: The AST node whose fields are to be iterated over.

    Yields:
    - A tuple containing the name of the field and the value of the field.
    """
    index = 0
    children = node.children()
    while index < len(children):
        name, child = children[index]
        try:
            bracket_index = name.index("[")
        except ValueError:
            yield name, child
            index += 1
        else:
            name = name[:bracket_index]
            child = getattr(node, name)
            index += len(child)
            yield name, child


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
                result = int(node.left.value) / int(node.right.value)
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


def simplify_code(source_code):
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


# 使用示例
source = """ 
void add(int* a, int* b) {
    for (int i_j_fuse = 0; i_j_fuse < 300 * 300; i_j_fuse++) {
        a[i_j_fuse] = b[i_j_fuse] + 4;
    }
}
"""

simplified_source = simplify_code(source)
print(simplified_source)
