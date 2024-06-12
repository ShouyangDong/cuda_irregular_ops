from pycparser import c_parser, c_ast

# C code with a function containing a for loop
c_code = """
int factorial(int result) {
    for (int i = 1; i <= 10; i++) {
        result *= i;
    }
    return result;
}
"""

# Custom visitor class to split 'for' loops with loop count 10 into 2 loops with counts 2 and 5
class SplitForLoopVisitor(c_ast.NodeVisitor):
    def visit_For(self, node):
        if isinstance(node.cond, c_ast.BinaryOp) and \
            isinstance(node.cond.left, c_ast.ID) and \
            isinstance(node.cond.op, '<=') and \
            isinstance(node.cond.right, c_ast.Constant) and \
            node.cond.right.value == '10':
            
            # Modify the existing 'for' loop to have loop count 5
            node.cond.right.value = '5'
            node.cond.op = '<'
            
            # Create a new 'for' loop with loop count 2
            new_for_loop = c_ast.For(init=node.init,
                                     cond=c_ast.BinaryOp('<', c_ast.ID('i'), c_ast.Constant('int', '3')),
                                     next=c_ast.UnaryOp('++', c_ast.ID('i')),
                                     stmt=node.stmt)

            # Replace the existing 'for' loop with the new one in the parent node
            parent_node = self.get_parent(node)
            stmt_index = parent_node.stmt.index(node)
            parent_node.stmt[stmt_index] = new_for_loop

    def get_parent(self, node):
        for parent, child in node.parent.items():
            if child == node:
                return parent
        return None

# Parse the C code
parser = c_parser.CParser()
ast = parser.parse(c_code)

# Custom visitor instance
visitor = SplitForLoopVisitor()

# Visit the AST to split 'for' loops with loop count 10 into 2 loops with counts 2 and 5
visitor.visit(ast)

# Generate the modified C code
modified_code = ast.show()

print(modified_code)