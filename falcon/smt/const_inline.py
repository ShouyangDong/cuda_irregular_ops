from pycparser import c_ast, c_generator, c_parser

from falcon.smt.util import NodeTransformer


class ConstInlineTransformer(NodeTransformer):
    def __init__(self):
        super().__init__()
        # 存储常量的字典
        self.constants = {}


class ConstInlineTransformer(NodeTransformer):
    def __init__(self):
        super().__init__()
        # 存储常量的字典
        self.constants = {}

    def visit_Decl(self, node):
        """找到并记录所有整数常量的值，并从 AST 中删除这些声明"""
        # 记录整数常量
        if isinstance(node.init, c_ast.Constant) and node.init.type == "int":
            self.constants[node.name] = node.init
            return None  # 返回 None 以删除该声明
        # 记录计算表达式
        if (
            isinstance(node.init, (c_ast.BinaryOp, c_ast.UnaryOp))
            and node.type.type.names[0] == "int"
        ):
            self.constants[node.name] = node.init
            return None  # 返回 None 以删除该声明
        return node

    def visit_ID(self, node):
        """将变量替换为对应的常量值或表达式"""
        # 用常量值或表达式替换变量
        if node.name in self.constants:
            return self.constants[node.name]
        return node

    def visit_DeclList(self, node):
        return node

    def visit_Assignment(self, node):
        """替换 'index' 的赋值表达式"""
        # 如果左值是 'index'，将整个赋值替换为右值
        if isinstance(node.lvalue, c_ast.ID) and node.lvalue.name in self.constants:
            return self.generic_visit(node.rvalue)
        return self.generic_visit(node)

    def visit_FuncDef(self, node):
        """在函数体中删除声明并替换计算表达式"""
        # 首先调用 visit 函数删除声明并替换变量
        self.generic_visit(node)
        # 移除已经不需要的声明（删除 None）
        node.body.block_items = [
            stmt for stmt in node.body.block_items if stmt is not None
        ]
        return node


def constant_inline(code):
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
    # 示例代码
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
