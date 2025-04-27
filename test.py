import re

from pycparser import CParser, c_ast, c_generator


def preprocess_cuda_for_cparser(code):
    # 替换 CUDA 线程索引
    mapping = {
        r"\bthreadIdx\.x\b": "__TIDX_X__",
        r"\bthreadIdx\.y\b": "__TIDX_Y__",
        r"\bthreadIdx\.z\b": "__TIDX_Z__",
        r"\bblockIdx\.x\b": "__BIDX_X__",
        r"\bblockIdx\.y\b": "__BIDX_Y__",
        r"\bblockIdx\.z\b": "__BIDX_Z__",
        r"\bblockDim\.x\b": "__BDIM_X__",
        r"\bblockDim\.y\b": "__BDIM_Y__",
        r"\bblockDim\.z\b": "__BDIM_Z__",
        r"\bgridDim\.x\b": "__GDIM_X__",
        r"\bgridDim\.y\b": "__GDIM_Y__",
        r"\bgridDim\.z\b": "__GDIM_Z__",
    }
    for pattern, replacement in mapping.items():
        code = re.sub(pattern, replacement, code)

    # CUDA 类型/函数打补丁
    typedefs = """
typedef float half;
typedef struct { float x[8]; } wmma_fragment;
void wmma_load_matrix_sync(wmma_fragment f, float *ptr, int stride);
"""

    # 替换 wmma:: -> wmma_
    code = re.sub(r"wmma::", "wmma_", code)

    # 去掉模板参数 <...>
    code = re.sub(r"<[^>]*>", "", code)

    # 去掉 __global__ __device__ __host__ 等
    code = re.sub(r"\b__global__\b", "", code)
    code = re.sub(r"\b__device__\b", "", code)
    code = re.sub(r"\b__host__\b", "", code)
    code = re.sub(r"\b__shared__\b", "", code)
    code = re.sub(r"\b__forceinline__\b", "", code)

    return typedefs + "\n" + code


def parse_cuda_as_c(code):
    preprocessed_code = preprocess_cuda_for_cparser(code)
    print(preprocessed_code)
    parser = CParser()
    ast = parser.parse(preprocessed_code)
    return ast


class TensorCoreDetensorizer(c_ast.NodeVisitor):
    def __init__(self):
        self.new_nodes = []

    def visit_FuncDef(self, node):
        self.generic_visit(node)
        print("[INFO]*******node: ", node)
        node.body.block_items = self.process_block(node.body.block_items)
        return node

    def process_block(self, block_items):
        if not block_items:
            return block_items

        new_block = []
        for stmt in block_items:
            if isinstance(stmt, c_ast.FuncCall):
                lowered = self.lower_tensorcore_func(stmt)
                if isinstance(lowered, list):
                    new_block.extend(lowered)
                else:
                    new_block.append(lowered)
            elif isinstance(
                stmt, (c_ast.If, c_ast.For, c_ast.While, c_ast.Compound)
            ):
                self.generic_visit(stmt)
                new_block.append(stmt)
            else:
                new_block.append(stmt)
        return new_block

    def lower_tensorcore_func(self, call):
        fname = self.get_func_name(call)

        if fname == "wmma_fill_fragment":
            frag = call.args.exprs[0]
            val = call.args.exprs[1]
            return self.generate_fill_fragment(frag, val)

        elif fname == "wmma_load_matrix_sync":
            frag = call.args.exprs[0]
            src_ptr = call.args.exprs[1]
            stride = call.args.exprs[2]
            return self.generate_load_fragment(frag, src_ptr, stride)

        elif fname == "wmma_store_matrix_sync":
            dst_ptr = call.args.exprs[0]
            frag = call.args.exprs[1]
            stride = call.args.exprs[2]
            return self.generate_store_fragment(dst_ptr, frag, stride)

        elif fname == "wmma_mma_sync":
            d_frag = call.args.exprs[0]
            a_frag = call.args.exprs[1]
            b_frag = call.args.exprs[2]
            c_frag = call.args.exprs[3]
            return self.generate_mma_sync(d_frag, a_frag, b_frag, c_frag)

        else:
            return call

    def get_func_name(self, call):
        if isinstance(call.name, c_ast.ID):
            return call.name.name
        elif isinstance(call.name, c_ast.StructRef):
            return call.name.field.name
        else:
            return None

    # ----------------具体展开指令----------------

    def generate_fill_fragment(self, frag, val):
        return c_ast.For(
            init=c_ast.Decl(
                name="i",
                quals=[],
                storage=[],
                funcspec=[],
                type=c_ast.TypeDecl(
                    declname="i",
                    quals=[],
                    align=None,
                    type=c_ast.IdentifierType(["int"]),
                ),
                init=c_ast.Constant("int", "0"),
                bitsize=None,
            ),
            cond=c_ast.BinaryOp(
                "<", c_ast.ID("i"), c_ast.Constant("int", "8")
            ),
            next=c_ast.UnaryOp("p++", c_ast.ID("i")),
            stmt=c_ast.Assignment(
                "=",
                c_ast.ArrayRef(
                    c_ast.StructRef(frag, "->", "x"), c_ast.ID("i")
                ),
                val,
            ),
        )

    def generate_load_fragment(self, frag, src_ptr, stride):
        return c_ast.For(
            init=c_ast.Decl(
                name="i",
                quals=[],
                storage=[],
                funcspec=[],
                type=c_ast.TypeDecl(
                    declname="i",
                    quals=[],
                    align=None,
                    type=c_ast.IdentifierType(["int"]),
                ),
                init=c_ast.Constant("int", "0"),
                bitsize=None,
                align=None,
            ),
            cond=c_ast.BinaryOp(
                "<", c_ast.ID("i"), c_ast.Constant("int", "8")
            ),
            next=c_ast.UnaryOp("p++", c_ast.ID("i")),
            stmt=c_ast.Assignment(
                "=",
                c_ast.ArrayRef(
                    c_ast.StructRef(frag, "->", "x"), c_ast.ID("i")
                ),
                c_ast.ArrayRef(src_ptr, c_ast.ID("i")),
            ),
        )

    def generate_store_fragment(self, dst_ptr, frag, stride):
        return c_ast.For(
            init=c_ast.Decl(
                "i",
                [],
                [],
                [],
                c_ast.TypeDecl("i", [], c_ast.IdentifierType(["int"])),
                c_ast.Constant("int", "0"),
            ),
            cond=c_ast.BinaryOp(
                "<", c_ast.ID("i"), c_ast.Constant("int", "8")
            ),
            next=c_ast.UnaryOp("p++", c_ast.ID("i")),
            stmt=c_ast.Assignment(
                "=",
                c_ast.ArrayRef(dst_ptr, c_ast.ID("i")),
                c_ast.ArrayRef(
                    c_ast.StructRef(frag, "->", "x"), c_ast.ID("i")
                ),
            ),
        )

    def generate_mma_sync(self, d_frag, a_frag, b_frag, c_frag):
        # 简单模拟 GEMM
        return [
            c_ast.For(
                init=c_ast.Decl(
                    "i",
                    [],
                    [],
                    [],
                    c_ast.TypeDecl("i", [], c_ast.IdentifierType(["int"])),
                    c_ast.Constant("int", "0"),
                ),
                cond=c_ast.BinaryOp(
                    "<", c_ast.ID("i"), c_ast.Constant("int", "8")
                ),
                next=c_ast.UnaryOp("p++", c_ast.ID("i")),
                stmt=c_ast.Assignment(
                    "=",
                    c_ast.ArrayRef(
                        c_ast.StructRef(d_frag, "->", "x"), c_ast.ID("i")
                    ),
                    c_ast.BinaryOp(
                        "+",
                        c_ast.BinaryOp(
                            "*",
                            c_ast.ArrayRef(
                                c_ast.StructRef(a_frag, "->", "x"),
                                c_ast.ID("i"),
                            ),
                            c_ast.ArrayRef(
                                c_ast.StructRef(b_frag, "->", "x"),
                                c_ast.ID("i"),
                            ),
                        ),
                        c_ast.ArrayRef(
                            c_ast.StructRef(c_frag, "->", "x"), c_ast.ID("i")
                        ),
                    ),
                ),
            )
        ]


# ----------使用方法-----------


def detensorize_tensorcore_ast(ast):
    detensorizer = TensorCoreDetensorizer()
    detensorizer.visit(ast)
    return ast


if __name__ == "__main__":
    cuda_code = """
    __global__ void kernel(half *A, half *B, float *C, float *D) {
        wmma::load_matrix_sync(a_frag, A + ix, 32);
    }
    """
    generator = c_generator.CGenerator()
    ast = parse_cuda_as_c(cuda_code)
    new_tmp = generator.visit(ast)
    print("[INFO]*****new_tmp: ", new_tmp)
    ast = detensorize_tensorcore_ast(ast)

    new_tmp = generator.visit(ast)
    print(new_tmp)
