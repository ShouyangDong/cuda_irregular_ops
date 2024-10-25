from pycparser import c_ast, c_parser

from smt.util import NodeTransformer


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
