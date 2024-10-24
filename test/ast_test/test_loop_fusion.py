from pycparser import c_ast, c_generator, c_parser

# C code with a function containing a for loop
c_code = """
void factorial(int* result) {
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            result[i * 10 + j] += 1;
        }
    }
}
"""


# Custom visitor class to reorder loop index i and j
class LoopReorderVisitor(c_ast.NodeVisitor):
    def __init__(self, axis_name_1, axis_name_2):
        self.axis_name_1 = axis_name_1
        self.axis_name_2 = axis_name_2
        self.extend = {}

    def visit_For(self, node):
        # check the loop index
        if node.init.decls[0].name == self.axis_name_1:
            compound_nested_node = node.stmt
            generator = c_generator.CGenerator()
            if isinstance(compound_nested_node, c_ast.Compound) and isinstance(
                compound_nested_node.block_items[0], c_ast.For
            ):
                nested_node = compound_nested_node.block_items[0]
                if nested_node.init.decls[0].name == self.axis_name_2:
                    self.extend[self.axis_name_2] = nested_node.cond.right.value
                    extend = int(node.cond.right.value) * int(
                        nested_node.cond.right.value
                    )
                    node.init.decls[0].name = (
                        "fuse_" + self.axis_name_1 + "_" + self.axis_name_2
                    )
                    node.init.decls[0].type.declname = (
                        "fuse_" + self.axis_name_1 + "_" + self.axis_name_2
                    )
                    node.cond.left.name = (
                        "fuse_" + self.axis_name_1 + "_" + self.axis_name_2
                    )
                    node.cond.right.value = extend
                    node.next.expr.name = (
                        "fuse_" + self.axis_name_1 + "_" + self.axis_name_2
                    )
                    # replace the loop index with new loop index
                    node.stmt = nested_node.stmt
                    self.visit(node.stmt)

    def visit_BinaryOp(self, node):
        if isinstance(node.left, c_ast.BinaryOp) and node.op == "+":
            if (
                node.left.op == "*"
                and node.left.left.name == self.axis_name_1
                and node.left.right.value == str(self.extend[self.axis_name_2])
                and node.right.name == self.axis_name_2
            ):
                node.left = c_ast.Constant(
                    "int", "fuse_" + self.axis_name_1 + "_" + self.axis_name_2
                )
                node.right = c_ast.Constant("int", 0)


# Parse the C code
parser = c_parser.CParser()
ast = parser.parse(c_code)
generator = c_generator.CGenerator()
print(generator.visit(ast))
# Custom visitor instance
visitor = LoopReorderVisitor("i", "j")

# Visit the AST to split 'for' loops with loop count 10 into 2 loops with counts 2 and 5
visitor.visit(ast)
print(generator.visit(ast))
