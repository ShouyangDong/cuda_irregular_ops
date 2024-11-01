from pycparser import c_ast, c_generator, c_parser

from smt.util import NodeTransformer


class CacheTransformationVisitor(NodeTransformer):
    def __init__(self, space_map):
        super().__init__()
        self.space_map = space_map  # 用于指定缓存的位置，如 {"A": "Nram", "B": "Nram"}

    def visit_FuncDef(self, node):
        """在函数定义节点内创建缓存缓冲区，并添加缓存加载和写回逻辑"""
        self.create_cache_buffers(node)  # 在函数开头创建缓存缓冲区
        return node

    def create_cache_buffers(self, node):
        """根据 space_map 创建 NRAM 缓冲区"""
        size_param = c_ast.ID(name="size")
        declarations = []

        # 根据 space_map 中的 input 和 output 动态创建 NRAM 缓冲区
        for mapping in self.space_map:
            for var_name, location in mapping.get("input", {}).items():
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
                            type=c_ast.IdentifierType(["float"]),
                        ),
                        dim=size_param,
                        dim_quals=[],
                    ),
                    align=None,
                    init=None,
                    bitsize=None,
                )
                declarations.append(nram_decl)

            for var_name, location in mapping.get("output", {}).items():
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
                            type=c_ast.IdentifierType(["float"]),
                        ),
                        dim=size_param,
                        dim_quals=[],
                    ),
                    align=None,
                    init=None,
                    bitsize=None,
                )
                declarations.append(nram_decl)

        # 插入到函数体的开头
        node.body.block_items = declarations + node.body.block_items
        self.generic_visit(node)

    def visit_Compound(self, node):
        """在找到 for 循环后插入缓存读写操作"""
        new_block_items = []
        start_cache = False
        for item in node.block_items:
            if isinstance(item, c_ast.Pragma) and "__bang_add" in item.string:
                # 检测到 #pragma 行，表示需要进行缓存操作
                start_cache = True
            elif isinstance(item, c_ast.For) and start_cache:
                # 插入缓存读取（读操作）
                read_items = self.insert_cache_read_operations(item)
                # 插入原始的 for 循环
                new_block_items.extend(read_items)
                # 修改循环体中的变量
                new_item = self.modify_for_loop_body(item)
                new_block_items.append(new_item)
                # 插入缓存写回（写操作）
                write_items = self.insert_cache_write_operations(item)
                new_block_items.extend(write_items)
                start_cache = False  # 重置标志
            else:
                new_block_items.append(item)
        node.block_items = new_block_items
        return node

    def modify_for_loop_body(self, for_node):
        """将 for 循环体内的变量替换为 NRAM 缓冲区变量"""
        for stmt in for_node.stmt.block_items:
            if isinstance(stmt, c_ast.Assignment):
                # 替换左值（例如 C[i_add] -> C_Nram[i_add]）
                if (
                    isinstance(stmt.lvalue, c_ast.ArrayRef)
                    and stmt.lvalue.name.name in self.space_map[0]["output"]
                ):
                    stmt.lvalue.name.name += (
                        "_" + self.space_map[0]["output"][stmt.lvalue.name.name]
                    )

                # 替换右值的输入变量（例如 A[i_add], B[i_add] -> A_Nram[i_add], B_Nram[i_add]）
                if isinstance(stmt.rvalue, c_ast.BinaryOp):
                    for operand in ["left", "right"]:
                        if isinstance(getattr(stmt.rvalue, operand), c_ast.ArrayRef):
                            array_ref = getattr(stmt.rvalue, operand)
                            if array_ref.name.name in self.space_map[0]["input"]:
                                array_ref.name.name += (
                                    "_"
                                    + self.space_map[0]["input"][array_ref.name.name]
                                )
        return for_node

    def insert_cache_read_operations(self, for_node):
        """在 for 循环前插入缓存读取逻辑"""
        read_operations = []
        # 遍历 space_map 中的 input 字段并生成加载操作
        for mapping in self.space_map:
            for var_name, location in mapping.get("input", {}).items():
                read_op = self.create_load_loop(
                    var_name, f"{var_name}_{location}", for_node
                )
                read_operations.append(read_op)
        return read_operations

    def insert_cache_write_operations(self, for_node):
        """在 for 循环后插入缓存写回逻辑"""
        write_operations = []

        # 遍历 space_map 中的 output 字段并生成写回操作
        for mapping in self.space_map:
            for var_name, location in mapping.get("output", {}).items():
                write_op = self.create_write_back_loop(
                    f"{var_name}_{location}", var_name, for_node
                )
                write_operations.append(write_op)

        # 将缓存写回操作插入到 for 循环之后
        return write_operations

    def create_load_loop(self, src, dest, for_node):
        """生成从 src 加载到 dest 的 for 循环"""
        index = c_ast.ID(name=for_node.init.decls[0].name)
        return c_ast.For(
            init=for_node.init,
            cond=for_node.cond,
            next=for_node.cond,
            stmt=c_ast.Compound(
                block_items=[
                    c_ast.Assignment(
                        op="=",
                        lvalue=c_ast.ArrayRef(
                            name=c_ast.ID(name=dest), subscript=index
                        ),
                        rvalue=c_ast.ArrayRef(name=c_ast.ID(name=src), subscript=index),
                    )
                ]
            ),
        )

    def create_write_back_loop(self, src, dest, for_node):
        """生成从 src 写回到 dest 的 for 循环"""
        index = c_ast.ID(name=for_node.init.decls[0].name)
        return c_ast.For(
            init=for_node.init,
            cond=for_node.cond,
            next=for_node.cond,
            stmt=c_ast.Compound(
                block_items=[
                    c_ast.Assignment(
                        op="=",
                        lvalue=c_ast.ArrayRef(
                            name=c_ast.ID(name=dest), subscript=index
                        ),
                        rvalue=c_ast.ArrayRef(name=c_ast.ID(name=src), subscript=index),
                    )
                ]
            ),
        )


def ast_auto_cache(code, space_map):
    # 解析代码
    print("[INFO]ast_auto_cache:", code)
    parser = c_parser.CParser()
    ast = parser.parse(code)
    # 进行缓存加载和写回插入
    transformer = CacheTransformationVisitor(space_map)
    ast = transformer.visit(ast)

    # 输出最终代码
    generator = c_generator.CGenerator()
    return generator.visit(ast)


if __name__ == "__main__":
    # 示例代码和 space_map
    code = """
    void __bang_add(float *C, float *A, float *B, int size) {
        #pragma __bang_add(input[Nram, Nram], output[Nram])
        for (int i_add = 0; i_add < size; i_add++) {
            C[i_add] = A[i_add] + B[i_add];
        }
    }
    """

    space_map = [{"input": {"A": "Nram", "B": "Nram"}, "output": {"C": "Nram"}}]
    output_code = ast_auto_cache(code, space_map)

    print(output_code)
