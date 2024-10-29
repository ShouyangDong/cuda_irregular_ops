from pycparser import c_ast, c_generator, c_parser

from smt.util import NodeTransformer



class CacheTransformationVisitor(NodeTransformer):
    def __init__(self, space_map):
        super().__init__()
        self.space_map = space_map

    def visit_FuncDef(self, node):
        """访问函数定义节点并修改其中的代码结构"""
        # 提取 for 循环，进行 transformation
        self.create_cache_buffers(node)
        return node

    def create_cache_buffers(self, node):
        """根据 space_maps 创建 NRAM 缓冲区并插入到函数开头"""
        size_param = c_ast.ID(name='size')
        declarations = []

        # 遍历 space_maps 中的 input 和 output 字段
        for mapping in self.space_map:
            # 处理 input 字段中的变量
            for var_name, location in mapping.get("input", {}).items():
                # 创建相应的 NRAM 缓冲区
                nram_decl = c_ast.Decl(
                    name=f"{var_name}_{location}",
                    quals=[],
                    storage=[],
                    funcspec=[],
                    type=c_ast.ArrayDecl(
                        type=c_ast.TypeDecl(
                            declname=f"{var_name}_{location}",
                            quals=[],
                            align=None,
                            type=c_ast.IdentifierType(['float'])
                        ),
                        dim=size_param,
                        dim_quals=[]
                    ),
                    align=None,
                    init=None,
                    bitsize=None
                )
                declarations.append(nram_decl)

            # 处理 output 字段中的变量
            for var_name, location in mapping.get("output", {}).items():
                # 创建相应的 NRAM 缓冲区
                nram_decl = c_ast.Decl(
                    name=f"{var_name}_{location}",
                    quals=[],
                    storage=[],
                    funcspec=[],
                    type=c_ast.ArrayDecl(
                        type=c_ast.TypeDecl(
                            declname=f"{var_name}_{location}",
                            quals=[],
                            align=None,
                            type=c_ast.IdentifierType(['float'])
                        ),
                        dim=size_param,
                        dim_quals=[]
                    ),
                    align=None,
                    init=None,
                    bitsize=None
                )
                declarations.append(nram_decl)

        # 将所有生成的缓冲区声明插入到函数体的开头
        node.body.block_items = declarations + node.body.block_items

    def visit_Compound(self, node):
        start_cache = False
        new_block_items = []
        for item in node.block_items:
            if isinstance(item, c_ast.Pragma) and "__bang_add" in item.string:
                start_cache = True
            elif isinstance(item, c_ast.For):
                # 添加加载缓存和计算逻辑
                self.insert_cache_operations(item, node)
                new_block_items.append(item)
            else:
                new_block_items.append(item)

        node.body.block_items = new_block_items
        return node


    # def insert_cache_operations(self, for_node, func_node):
    #     """插入缓存加载和写回逻辑"""
    #     size_id = c_ast.ID(name='size')

    #     # 从 A 加载到 A_nram
    #     load_A_nram = self.create_load_loop("A", "A_nram", size_id)
    #     func_node.body.block_items.insert(3, load_A_nram)

    #     # 从 B 加载到 B_nram
    #     load_B_nram = self.create_load_loop("B", "B_nram", size_id)
    #     func_node.body.block_items.insert(4, load_B_nram)

    #     # 从 C_nram 写回到 C
    #     write_C = self.create_write_back_loop("C_nram", "C", size_id)
    #     func_node.body.block_items.append(write_C)

    # def create_load_loop(self, src, dest, size):
    #     """生成从 src 加载到 dest 的循环"""
    #     index = c_ast.ID(name="i_add")
    #     return c_ast.For(
    #         init=c_ast.Decl(name="i_add", quals=[], storage=[], type=c_ast.TypeDecl(declname="i_add", quals=[], align=None, type=c_ast.IdentifierType(['int'])), init=c_ast.Constant(type="int", value="0")),
    #         cond=c_ast.BinaryOp(op="<", left=index, right=size),
    #         next=c_ast.UnaryOp(op="p++", expr=index),
    #         stmt=c_ast.Compound(block_items=[
    #             c_ast.Assignment(op="=", lvalue=c_ast.ArrayRef(name=c_ast.ID(name=dest), subscript=index),
    #                              rvalue=c_ast.ArrayRef(name=c_ast.ID(name=src), subscript=index))
    #         ])
    #     )

    # def create_write_back_loop(self, src, dest, size):
    #     """生成从 src 写回到 dest 的循环"""
    #     index = c_ast.ID(name="i_add")
    #     return c_ast.For(
    #         init=c_ast.Decl(name="i_add", quals=[], storage=[], type=c_ast.TypeDecl(declname="i_add", quals=[], align=None, type=c_ast.IdentifierType(['int'])), init=c_ast.Constant(type="int", value="0")),
    #         cond=c_ast.BinaryOp(op="<", left=index, right=size),
    #         next=c_ast.UnaryOp(op="p++", expr=index),
    #         stmt=c_ast.Compound(block_items=[
    #             c_ast.Assignment(op="=", lvalue=c_ast.ArrayRef(name=c_ast.ID(name=dest), subscript=index),
    #                              rvalue=c_ast.ArrayRef(name=c_ast.ID(name=src), subscript=index))
    #         ])
    #     )






if __name__ == "__main__":
    # 示例代码和 space maps
    code = """
        void add(C, A, B, size) {
            #pragma __bang_add(input[Nram, Nram], output[Nram])
            for (int i_add = 0; i_add < size; i_add++) {
                C[i_add] = A[i_add] + B[i_add];
            }
        }
    """
    space_maps = [
        {"input": {"A": "Nram", "B": "Nram"}, "output": {"C": "Nram"}},
    ]

    # 解析代码
    parser = c_parser.CParser()
    ast = parser.parse(code)

    # 进行 cache transformation
    transformer = CacheTransformationVisitor(space_maps)
    ast = transformer.visit(ast)

    # 生成最终代码
    generator = c_generator.CGenerator()
    output_code = generator.visit(ast)
    print(output_code)
