from pycparser import c_ast, c_generator, c_parser

# C code with a function containing a for loop
c_code = """
int factorial(int result) {
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            result += i + j;
        }
    }
    return result;
}
"""


# Custom visitor class to reorder loop index i and j
class LoopReorderVisitor(c_ast.NodeVisitor):
    def __init__(self, axis_name_1, axis_name_2):
        self.axis_name_1 = axis_name_1
        self.axis_name_2 = axis_name_2

    def visit_For(self, node):
        # check the loop index
        if node.init.decls[0].name == self.axis_name_1:
            compound_nested_node = node.stmt
            c_generator.CGenerator()
            if isinstance(compound_nested_node, c_ast.Compound) and isinstance(
                compound_nested_node.block_items[0], c_ast.For
            ):
                nested_node = compound_nested_node.block_items[0]
                if nested_node.init.decls[0].name == self.axis_name_2:
                    stmt_node = nested_node.stmt
                    inner_loop = c_ast.For(
                        init=node.init,
                        cond=node.cond,
                        next=node.next,
                        stmt=stmt_node,
                    )
                    node.init = nested_node.init
                    node.cond = nested_node.cond
                    node.next = nested_node.next
                    node.stmt = c_ast.Compound(block_items=[inner_loop])


# Parse the C code
parser = c_parser.CParser()
ast = parser.parse(c_code)
generator = c_generator.CGenerator()
print(generator.visit(ast))
# Custom visitor instance
visitor = LoopReorderVisitor("i", "j")

# Visit the AST to split 'for' loops with loop count 10 into 2 loops with
# counts 2 and 5
visitor.visit(ast)
print(generator.visit(ast))
