from pycparser import c_parser, c_ast, c_generator


class NodeTransformer(c_ast.NodeVisitor):
    def generic_visit(self, node):
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
    # this doesn't look pretty because `pycparser` decided to have structure
    # for AST node classes different from stdlib ones
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

class CacheWriteTransformer(NodeTransformer):
    def __init__(self, buffer_var, axis, pragma_info, args):
        self.buffer_var = buffer_var
        self.axis = axis
        self.pragma_info = pragma_info
        self.agrs = args

    def visit_Pragma(self, node):
        return None

    def visit_For(self, node):
        if node.init.decls[0].name == self.axis
            pragam = self.pragma_info[node]


        else:
            return node

class PragmaVisitor(c_ast.NodeVisitor):
    def __init__(self):
        self.pragma_info = {}
        self.agrs = {}

    def visit_Compound(self, node):
        # Get the block_items
        blocks = node.block_items
        for index, node in enumerate(blocks):
            if isinstance(node, c_ast.Pragma):
                self.pragma_info[blocks[index + 1]] = node.string

    def visit_BinaryOp(self, node):


if __name__ == "__main__":
    code = """
        void __bang_add(C, A, B, size) {
            #pragma __bang_add(input[Nram, Nram], output[Nram])
            for (int i_add = 0; i_add < size; i_add++) {
                C[i_add] = A[i_add] + B[i_add];
            }
        }
    """