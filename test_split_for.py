from pycparser import c_parser, c_ast, c_generator

# C code with a function containing a for loop
c_code = """
int factorial(int result) {
    for (int i = 0; i <= 10; i++) {
        result *= i;
    }
    return result;
}
"""

# Custom visitor class to split 'for' loops with loop count 10 into 2 loops with counts 2 and 5
class SplitForLoopVisitor(c_ast.NodeVisitor):
    def __init__(self, axis_name):
        self.axis_name = axis_name

    def visit_For(self, node):
        # get the loop index:
        # print(node.init.decls[0].name)
        if node.init.decls[0].name == self.axis_name:
            node.cond.right.value = '5'
            self.visit(node.stmt)

    def visit_ID(self, node):
        if node.name == self.axis_name:
            node.name = self.axis_name + "_out"

# Parse the C code
parser = c_parser.CParser()
ast = parser.parse(c_code)
generator = c_generator.CGenerator()
print(generator.visit(ast))
# Custom visitor instance
visitor = SplitForLoopVisitor("i")

# Visit the AST to split 'for' loops with loop count 10 into 2 loops with counts 2 and 5
visitor.visit(ast)

# # Generate the modified C code
# ast.show()
print(generator.visit(ast))