from pycparser import c_ast, c_generator, c_parser


class PragmaVisitor(c_ast.NodeVisitor):
    def __init__(self):
        self.pragma_info = {}

    def visit_Compound(self, node):
        # Get the block_items
        blocks = node.block_items
        for index, node in enumerate(blocks):
            if isinstance(node, c_ast.Pragma):
                self.pragma_info[node.string] = blocks[index + 1]


if __name__ == "__main__":
    c_code = """
    void matrix_multiply(int dest[SIZE][SIZE], int src1[SIZE][SIZE], int src2[SIZE][SIZE]) {
        #pragma operation(matmul)
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                for (int k = 0; k < SIZE; k++) {
                    dest[i][j] += src1[i][k] * src2[k][j];
                }
            }
        }
    }
    """
    parser = c_parser.CParser()
    ast = parser.parse(c_code)
    generator = c_generator.CGenerator()
    print(generator.visit(ast))
    # Custom visitor instance
    visitor = PragmaVisitor()
    visitor.visit(ast)
    for key, value in visitor.pragma_info.items():
        print("[INFO]***********key: ", key)
        print("[INFO]*********value: ", generator.visit(value))

    c_code = """
    void matrix_multiply(int dest[SIZE][SIZE], int src1[SIZE][SIZE], int src2[SIZE][SIZE]) {
        #pragma intrinsic(__bang_mlp(input[Nram, Wram], output[Nram]))
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                for (int k = 0; k < SIZE; k++) {
                    dest[i][j] += src1[i][k] * src2[k][j];
                }
            }
        }
    }
    """
    parser = c_parser.CParser()
    ast = parser.parse(c_code)
    generator = c_generator.CGenerator()
    print(generator.visit(ast))
    # Custom visitor instance
    visitor = PragmaVisitor()
    visitor.visit(ast)
    for key, value in visitor.pragma_info.items():
        print("[INFO]***********key: ", key)
        print("[INFO]*********value: ", generator.visit(value))
