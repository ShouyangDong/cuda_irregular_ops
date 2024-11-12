from pycparser import c_ast, c_generator, c_parser

from falcon.smt.util import NodeTransformer


class LoopSplitter(NodeTransformer):
    def visit_Compound(self, node):
        # Split each for-loop with multiple statements into separate loops
        new_block_items = []
        for stmt in node.block_items:
            if isinstance(stmt, c_ast.For):
                # Check if the loop body has multiple statements
                if (
                    isinstance(stmt.stmt, c_ast.Compound)
                    and len(stmt.stmt.block_items) > 1
                ):
                    # Create a new `for` loop for each statement in the original loop body
                    for single_stmt in stmt.stmt.block_items:
                        new_for = c_ast.For(
                            init=stmt.init,
                            cond=stmt.cond,
                            next=stmt.next,
                            stmt=c_ast.Compound(
                                [single_stmt]
                            ),  # Single statement loop body
                        )
                        new_block_items.append(new_for)
                else:
                    # If the loop has only one statement, add it as is
                    new_block_items.append(stmt)
            else:
                # Non-loop statements are added unchanged
                new_block_items.append(stmt)

        # Update block items with the split loops
        node.block_items = new_block_items
        return node


# Sample code to transform
code = """
void sum(float* expf, float* T_softmax_maxelem) {
float denom = 0.0f;
float maxVal = -3.0f;
for (int i = 0; i < 5; ++i) {
    T_softmax_maxelem[threadIdxx * 5 + i] = expf(A[threadIdxx * 5 + i] - maxVal);
    denom += T_softmax_maxelem[threadIdxx * 5 + i];
}
}
"""

# Parse code and apply loop splitting
parser = c_parser.CParser()
ast = parser.parse(code)

# Apply loop splitting transformation
splitter = LoopSplitter()
split_ast = splitter.visit(ast)

# Generate and print transformed code
generator = c_generator.CGenerator()
transformed_code = generator.visit(split_ast)
print(transformed_code)
