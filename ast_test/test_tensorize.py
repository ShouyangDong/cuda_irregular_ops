from pycparser import c_parser, c_ast, c_generator


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


class PragmaVisitor(c_ast.NodeVisitor):
    def __init__(self):
        self.pragma_info = {}

    def visit_Compound(self, node):
        # Get the block_items
        blocks = node.block_items
        for index, node in enumerate(blocks):
            if isinstance(node, c_ast.Pragma):
                self.pragma_info[node.string] = blocks[index + 1]


class TensorizationTransformer(NodeTransformer):
    def __init__(self, pragma_info):
        self.pragma_info = pragma_info

    def visit_Pragma(self, node):
        return None

    def visit_For(self, node):
        # Check the node in prama information
        # Check node in pragma info
        for pragma_node, pragma_str in self.pragma_info.items():
            if not are_nodes_equal(node, pragma_node):
                return node

            else:
                # Tensorize the for loop
                code = postprocess(node)
                intrinsic = metalift(code, pragma)
                final_node = stitch_code(intrinsic)
                return final_node

    def postprocess(code):
        return code


if __name__ == "__main__":
    code = """
    void __bang_add(C, A, B, size) {
        #pragma operation(add)
        for (int i_add = 0; i_add < size; i_add++) {
            C[i_add] = A[i_add] + B[i_add];
        }
    }"
    """
    parser = c_parser.CParser()
    ast = parser.parse(code)
    visitor = PragmaVisitor()
    visitor.visit(ast)
    tensorizer = TensorizationTransformer(visitor.pragma_info)
    tensorizer.visit(ast)
