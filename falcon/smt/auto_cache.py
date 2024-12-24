from pycparser import c_ast, c_generator, c_parser

from falcon.smt.util import (
    NodeTransformer,
    add_memory_prefix,
    remove_target_prefix,
)


class LoopVisitor(c_ast.NodeVisitor):
    def __init__(self):
        # self.cache_node = {}
        self.cache_size = []

    def visit_Compound(self, node):
        start_cache = False
        for item in node.block_items:
            if isinstance(item, c_ast.Pragma) and "__bang" in item.string:
                # 检测到 #pragma 行，表示需要进行缓存操作
                start_cache = True
                # self.cache_node[item.string] = None
            elif isinstance(item, c_ast.For) and start_cache:
                # self.cache_node[item.string] = item
                self.cache_size.append(item.cond.right.value)
                start_cache = False  # 重置标志
        self.generic_visit(node)


class CacheTransformationVisitor(NodeTransformer):
    def __init__(self, space_map, cache_size):
        super().__init__()
        self.space_map = space_map
        self.cache_size = cache_size

    def visit_FuncDef(self, node):
        """在函数定义节点内创建缓存缓冲区，并添加缓存加载和写回逻辑"""
        self.create_cache_buffers(node)  # 在函数开头创建缓存缓冲区
        return node

    def create_cache_buffers(self, node):
        """根据 space_map 创建 NRAM 缓冲区"""
        size_param = c_ast.Constant(type="int", value=self.cache_size[0])
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
            if isinstance(item, c_ast.Pragma) and "__bang" in item.string:
                # 检测到 #pragma 行，表示需要进行缓存操作
                start_cache = True
            elif isinstance(item, c_ast.For) and start_cache:
                reads, writes = self.extract_index_expression(item)
                # 插入缓存读取（读操作）
                read_items = self.create_read_operations(item, reads)
                # 插入原始的 for 循环
                new_block_items.extend(read_items)
                # 修改循环体中的变量
                new_item = self.modify_for_loop_body(item, reads, writes)
                new_block_items.append(new_item)
                # 插入缓存写回（写操作）
                write_items = self.create_write_operations(item, writes)
                new_block_items.extend(write_items)
                start_cache = False  # 重置标志
            else:
                new_block_items.append(item)
        node.block_items = new_block_items
        return self.generic_visit(node)

    def modify_for_loop_body(self, for_node, reads, writes):
        """将 for 循环体内的变量替换为 NRAM 缓冲区变量"""
        index = c_ast.ID(name=for_node.init.decls[0].name)
        inputs = []
        for var_name, location in self.space_map[0]["input"].items():
            inputs.append(
                c_ast.ArrayRef(
                    name=c_ast.ID(name=f"{var_name}_{location}"),
                    subscript=index,
                )
            )
        ouptuts = []
        for var_name, location in self.space_map[0]["output"].items():
            ouptuts.append(
                c_ast.ArrayRef(
                    name=c_ast.ID(name=f"{var_name}_{location}"),
                    subscript=index,
                )
            )

        stmt = for_node.stmt.block_items[0]
        assert isinstance(stmt, c_ast.Assignment)
        right = stmt.rvalue
        left_value = ouptuts[0]
        right_value = None
        if isinstance(right, c_ast.BinaryOp):
            right_value = c_ast.BinaryOp(
                op=right.op, left=inputs[0], right=inputs[1]
            )

        final_node = c_ast.For(
            init=for_node.init,
            cond=for_node.cond,
            next=for_node.next,
            stmt=c_ast.Compound(
                block_items=[
                    c_ast.Assignment(
                        op="=", lvalue=left_value, rvalue=right_value
                    )
                ]
            ),
        )

        return final_node

    def extract_index_expression(self, for_node):
        src_index = {}
        stmt = for_node.stmt.block_items[0]
        assert isinstance(stmt, c_ast.Assignment)
        right = stmt.rvalue
        if isinstance(right, c_ast.BinaryOp):
            src_index[right.left.name.name] = right.left
            src_index[right.right.name.name] = right.right
        return src_index, stmt.lvalue

    def create_read_operations(self, for_loop, src_index):
        """Insert cache read operations with complex indexing."""
        reads = []
        for var_name, location in self.space_map[0]["input"].items():
            reads.append(
                self.create_load_loop(
                    var_name, f"{var_name}_{location}", for_loop, src_index
                )
            )
        return reads

    def create_write_operations(self, for_loop, dest_index):
        """Insert cache write-back operations with complex indexing."""
        writes = []
        for var_name, location in self.space_map[0]["output"].items():
            writes.append(
                self.create_write_back_loop(
                    f"{var_name}_{location}", for_loop, dest_index
                )
            )
        return writes

    def create_load_loop(self, src, dest, for_node, src_index):
        """Creates a load loop with specified complex index expression."""
        index = c_ast.ID(name=for_node.init.decls[0].name)
        return c_ast.For(
            init=for_node.init,
            cond=for_node.cond,
            next=for_node.next,
            stmt=c_ast.Compound(
                block_items=[
                    c_ast.Assignment(
                        op="=",
                        lvalue=c_ast.ArrayRef(
                            name=c_ast.ID(name=dest), subscript=index
                        ),
                        rvalue=src_index[src],
                    )
                ]
            ),
        )

    def create_write_back_loop(self, src, for_node, index_expr):
        """Creates a write-back loop with specified complex index expression."""
        index = c_ast.ID(name=for_node.init.decls[0].name)
        return c_ast.For(
            init=for_node.init,
            cond=for_node.cond,
            next=for_node.next,
            stmt=c_ast.Compound(
                block_items=[
                    c_ast.Assignment(
                        op="=",
                        lvalue=index_expr,
                        rvalue=c_ast.ArrayRef(
                            name=c_ast.ID(name=src), subscript=index
                        ),
                    )
                ]
            ),
        )


def ast_auto_cache(code, space_map, target="BANG"):
    code = remove_target_prefix(code, target)

    # 解析代码
    parser = c_parser.CParser()
    ast = parser.parse(code)
    # 进行缓存加载和写回插入
    cache_visitor = LoopVisitor()
    cache_visitor.visit(ast)
    transformer = CacheTransformationVisitor(
        space_map, cache_visitor.cache_size
    )
    ast = transformer.visit(ast)

    # 输出最终代码
    generator = c_generator.CGenerator()
    cache_code = generator.visit(ast)
    if target == "BANG":
        return add_memory_prefix(cache_code)
    else:
        return "__global__ " + cache_code


if __name__ == "__main__":
    # 示例代码和 space_map
    code = """
    void __bang_add(float *C, float *A, float *B) {
        #pragma __bang_add(input[Nram, Nram], output[Nram])
        for (int i_add = 0; i_add < 128; i_add++) {
            C[i_add] = A[i_add] + B[i_add];
        }
    }
    """

    space_map = [
        {"input": {"A": "Nram", "B": "Nram"}, "output": {"C": "Nram"}}
    ]
    output_code = ast_auto_cache(code, space_map)

    print(output_code)
