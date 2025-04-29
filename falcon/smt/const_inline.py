import copy

from pycparser import c_ast, c_generator, c_parser

from falcon.util import NodeTransformer, remove_target_prefix


class ConstInlineTransformer(NodeTransformer):
    def __init__(self):
        super().__init__()
        # 存储常量或简单表达式的映射：var_name -> AST node
        self.constants = {}

    def visit_Decl(self, node):
        # 只处理纯常量初始化（如 int dim = 3;），不内联复杂表达式
        if node.init and isinstance(node.init, c_ast.Constant):
            if isinstance(node.type, c_ast.TypeDecl):
                tnames = node.type.type.names
                if "int" in tnames or "float" in tnames:
                    # 记录常量并删除声明
                    self.constants[node.name] = copy.deepcopy(node.init)
                    return None
        return self.generic_visit(node)

    def visit_Assignment(self, node):
        # 只记录简单赋值（如 x = CONSTANT;），不删除复杂赋值
        if (
            isinstance(node.lvalue, c_ast.ID)
            and isinstance(node.rvalue, c_ast.Constant)
            and node.lvalue.name not in self.constants
        ):
            self.constants[node.lvalue.name] = copy.deepcopy(node.rvalue)
            return None
        # 其他情况递归替换
        node.lvalue = self.visit(node.lvalue)
        node.rvalue = self.visit(node.rvalue)
        return node

    def visit_For(self, node):
        # 处理 For 循环的初始化、条件和步进
        node.init = self.visit(node.init) if node.init else None
        node.cond = self.visit(node.cond) if node.cond else None
        node.next = self.visit(node.next) if node.next else None
        node.stmt = self.visit(node.stmt)
        return node

    def visit_ID(self, node):
        # 用常量表达式替换 ID
        if node.name in self.constants:
            return copy.deepcopy(self.constants[node.name])
        return node

    def visit_BinaryOp(self, node):
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)
        return node

    def visit_UnaryOp(self, node):
        node.expr = self.visit(node.expr)
        return node

    def visit_ArrayRef(self, node):
        # 处理 array[index] 中的索引表达式
        node.name = self.visit(node.name)
        node.subscript = self.visit(node.subscript)
        return node

    def visit_DeclList(self, node):
        # 清理空声明列表
        node.decls = [d for d in node.decls if d is not None]
        if not node.decls:
            return None
        return node

    def visit_FuncDef(self, node):
        # 先递归替换函数体内部节点
        self.generic_visit(node)
        # 清理因删除声明产生的空语句
        if node.body and node.body.block_items:
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
