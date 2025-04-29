from pycparser import c_ast, c_generator, c_parser

from falcon.util import NodeTransformer, remove_target_prefix


class ConstInlineTransformer(NodeTransformer):
    def __init__(self):
        super().__init__()
        self.constants = {}

    def visit_Decl(self, node):
        """Find and record integer constants or expressions assigned to ints,
           and remove these declarations from the AST."""
        if node.name and node.init and node.type and hasattr(node.type, 'type'):
            is_int_type = False
            if isinstance(node.type, c_ast.TypeDecl) and \
               isinstance(node.type.type, c_ast.IdentifierType) and \
               'int' in node.type.type.names:
                   is_int_type = True

            if is_int_type:
                if isinstance(node.init, c_ast.Constant) and node.init.type == "int":
                    # print(f"DEBUG: Found constant decl: {node.name} = {node.init.value}")
                    self.constants[node.name] = node.init
                    return None # Remove declaration

                elif isinstance(node.init, (c_ast.BinaryOp, c_ast.UnaryOp, c_ast.ID, c_ast.Constant)):
                    # print(f"DEBUG: Found constant expression decl: {node.name} = <expression>")
                    visited_init = self.visit(node.init)
                    self.constants[node.name] = visited_init
                    return None # Remove declaration

        return self.generic_visit(node) # Visit children even if not removed

    def visit_ID(self, node):
        """Replace variable IDs with their corresponding constant value or expression node.
           Crucially, recursively visit the replacement node."""
        if node.name in self.constants:
            replacement_node = self.constants[node.name]
            # print(f"DEBUG: Replacing ID '{node.name}' with node type {type(replacement_node)}")
            # Use deepcopy if there's a risk of modifying the stored node when
            # it's inserted multiple times. Usually safe for constants/simple exprs.
            # return self.visit(copy.deepcopy(replacement_node))
            return self.visit(replacement_node) # Recursive visit is key
        return node

    # ***** ADDED METHOD *****
    def visit_DeclList(self, node):
        """
        Visit a DeclList. If, after visiting the declarations within it,
        the list becomes empty (because all declarations were removed, e.g., constants),
        then remove the DeclList node itself to prevent CGenerator errors.
        """
        # First, let the generic visitor process the declarations in node.decls.
        self.generic_visit(node)
        # Now, check if the list of declarations is empty.
        if not node.decls:
            # print("DEBUG: Removing empty DeclList node.")
            return None # Remove this empty DeclList node
        else:
            return node # Keep the non-empty DeclList

    def visit_FuncDef(self, node):
        """Visit function body and ensure None statements are removed."""
        self.generic_visit(node)
        if node.body and hasattr(node.body, 'block_items') and node.body.block_items is not None:
             node.body.block_items = [
                stmt for stmt in node.body.block_items if stmt is not None
             ]
        return node
    
def constant_inline(code):
    code = remove_target_prefix(code)
    # 解析代码
    parser = c_parser.CParser()
    ast = parser.parse(code)
    # 进行转换
    transformer = ConstInlineTransformer()
    ast = transformer.visit(ast)

    # 输出转换后的代码
    generator = c_generator.CGenerator()
    return generator.visit(ast)


if __name__ == "__main__":
    # # 示例代码
    code = """
    void add_kernel(float *input1, float *input2, float *output)
    {
    float input1_Nram[size];
    float input2_Nram[size];
    float C_Nram[size];
    int dim1 = 4;
    int dim2 = 4;
    int dim3 = 4;
    int dim4 = 64;
    for (int k = 0; k < dim3; k++)
    {
        for (int l = 0; l < dim4; l++)
        {
        int index = (((((clusterId * dim2) * dim3) * dim4) + ((coreId * dim3) * dim4)) + (k * dim4)) + l;
        #pragma intrinsic(__bang_add(input[Nram, Nram], output[Nram]))
        output[index] = input1[index] + input2[index];
        }
    }
    }
    """
    code = constant_inline(code)
    print(code)

    code = """
    void softmax(float *A, float *T_softmax_norm)
    {
    for (int threadIdxx = 0; threadIdxx < 5; ++threadIdxx)
    {
        int rowStart = threadIdxx * 128;
        float maxVal = A[rowStart];
        for (int i = 1; i < 128; ++i)
        {
        if (A[rowStart + i] > maxVal)
        {
            maxVal = A[rowStart + i];
        }
        }

        float denom = 0.0f;
        for (int i = 0; i < 128; ++i)
        {
        T_softmax_norm[rowStart + i] = expf();
        denom += T_softmax_norm[rowStart + i];
        }

        for (int i = 0; i < 128; ++i)
        {
        T_softmax_norm[rowStart + i] /= denom;
        }

    }

    }
    """
    code = constant_inline(code)
    print(code)
