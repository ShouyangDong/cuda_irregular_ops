from pycparser import c_ast, c_generator, c_parser

# C code with a function containing a for loop
c_code = """
int factorial(int result) {
    for (int i = 0; i < 10; i++) {
        result += i;
    }
    return result;
}
"""


# Custom visitor class to split 'for' loops with loop count 10 into 2 loops with counts 2 and 5
class SplitForLoopVisitor(c_ast.NodeVisitor):
    def __init__(self, axis_name, factor):
        self.axis_name = axis_name
        self.factor = factor

    def visit_For(self, node):
        # check the loop index
        if node.init.decls[0].name == self.axis_name:
            org_extent = int(node.cond.right.value)
            node.cond.right.value = self.factor
            self.visit(node.stmt)
            init_node = c_ast.Decl(
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
            cond_node = c_ast.BinaryOp(
                node.cond.op,
                c_ast.ID(self.axis_name + "_in"),
                c_ast.Constant("int", node.cond.right.value),
            )
            next_node = c_ast.UnaryOp(node.next.op, c_ast.ID(self.axis_name + "_in"))

            inner_loop = c_ast.For(
                init=init_node, cond=cond_node, next=next_node, stmt=node.stmt
            )
            inner_loop = c_ast.Compound(block_items=[inner_loop])
            node.init = c_ast.Decl(
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
            node.cond = c_ast.BinaryOp(
                node.cond.op,
                c_ast.ID(self.axis_name + "_out"),
                c_ast.Constant("int", str(org_extent // self.factor)),
            )
            node.next = c_ast.UnaryOp(node.next.op, c_ast.ID(self.axis_name + "_out"))
            node.stmt = inner_loop

    def visit_ID(self, node):
        # modify the aixs name inside stmt
        if node.name == self.axis_name:
            node.name = (
                self.axis_name
                + "_out"
                + " * "
                + str(self.factor)
                + " + "
                + self.axis_name
                + "_in"
            )


# Parse the C code
parser = c_parser.CParser()
ast = parser.parse(c_code)
generator = c_generator.CGenerator()
print(generator.visit(ast))
# Custom visitor instance
visitor = SplitForLoopVisitor("i", factor=2)

# Visit the AST to split 'for' loops with loop count 10 into 2 loops with counts 2 and 5
visitor.visit(ast)
print(generator.visit(ast))
