# from pycparser import c_ast, c_generator, c_parser

# from falcon.smt.simplification import simplify_code
# from falcon.util import NodeTransformer


# class FuncCallsRemover(NodeTransformer):
#     def __init__(self, func_defs):
#         self.func_defs = func_defs
#         self.parser = c_parser.CParser()
#         self.parameter_mappings = {}

#     def visit_FuncCall(self, node):
#         if node.name.name in self.func_defs:
#             func_def = self.func_defs[node.name.name]
#             seq_def = self.parser.parse(func_def)
#             if not isinstance(seq_def, c_ast.FileAST):
#                 raise ValueError("Sequential code must be a function")

#             # Construct a map between the function call's  arguments and
#             # callee's arguments
#             seq_def_args = seq_def.ext[0].decl.type.args.params
#             seq_def_name = [arg_id.name for arg_id in seq_def_args]
#             self.parameter_mappings = {
#                 arg: param for arg, param in zip(seq_def_name, node.args.exprs)
#             }
#             body = seq_def.ext[0].body
#             return self.visit(body.block_items[0])
#         else:
#             return node

#     def visit_ID(self, node):
#         if node.name in self.parameter_mappings:
#             return self.parameter_mappings[node.name]
#         return node


# if __name__ == "__main__":
#     code = """
#     void add(float* lhs, float* rhs, float* add_1515) {
#         float lhs_local_nram[128];
#         __memcpy(((float *)lhs_local_nram + (0)), ((float *)lhs + (((((int)clusterId) * 256) + (((int)coreId) * 64)))), 256, GDRAM2NRAM);
#         __memcpy(((float *)lhs_local_nram + (64)), ((float *)rhs + (((((int)clusterId) * 256) + (((int)coreId) * 64)))), 256, GDRAM2NRAM);
#         __bang_add(((float *)lhs_local_nram + (0)), ((float *)lhs_local_nram + (0)), ((float *)lhs_local_nram + (64)), 64);
#         __memcpy(((float *)add_1515 + (((((int)clusterId) * 256) + (((int)coreId) * 64)))), ((float *)lhs_local_nram + (0)), 256, NRAM2GDRAM);
#     }
#     """
#     func_map = {
#         "__memcpy": "void memcpy(float* dst, float* src, int size, char direction) {for(int i=0; i<size/4; i++) {dst[i]=src[i];}}",
#         "__bang_add": "void bang_add(float* C, float* A, float* B, int size) {for(int i=0; i < size; i++){C[i]=A[i] + B[i];}}",
#     }

import re

#     parser = c_parser.CParser()
#     ast = parser.parse(code)
#     v = FuncCallsRemover(func_map)
#     v.visit(ast)
#     generator = c_generator.CGenerator()
#     code = simplify_code(generator.visit(ast))
#     print(code)
from pycparser import CParser, c_ast, c_generator

from falcon.util import NodeTransformer


class TensorCoreToScalar(NodeTransformer):
    def __init__(self, M=16, N=16, K=16):
        super().__init__()
        self.M = M
        self.N = N
        self.K = K

    def visit_FuncCall(self, node):
        # 取得调用名，比如 "wmma::load_matrix_sync"
        fn = node.name
        if isinstance(fn, c_ast.ID):
            name = fn.name
        elif isinstance(fn, c_ast.StructRef):
            # 可能是 :: 运算，也可用 name = fn.field.name
            name = fn.field.name
        else:
            name = ""

        args = node.args.exprs

        # 1) load_matrix_sync(a_frag, A + base, stride)
        if name.endswith("load_matrix_sync"):
            frag, ptr, stride = args
            # 把 frag 当作一个局部 C 数组 a_frag.x[0..M*N-1]，
            # 将从全局内存加载改写成双层 loop：
            # for i in 0..M-1: for j in 0..K-1: a_frag.x[i*K+j] = <*(ptr +
            # i*stride + j)>
            loop_i = c_ast.For(
                init=c_ast.Decl(
                    name="i",
                    quals=[],
                    storage=[],
                    funcspec=[],
                    type=c_ast.Typename(
                        name="int",
                        type=c_ast.TypeDecl(
                            declname="i",
                            quals=[],
                            type=c_ast.IdentifierType(["int"]),
                        ),
                    ),
                    init=c_ast.Constant("int", "0"),
                    bitsize=None,
                ),
                cond=c_ast.BinaryOp(
                    "<", c_ast.ID("i"), c_ast.Constant("int", str(self.M))
                ),
                next=c_ast.UnaryOp("p++", c_ast.ID("i")),
                stmt=c_ast.Compound(
                    [
                        c_ast.For(
                            init=c_ast.Decl(
                                name="j",
                                quals=[],
                                storage=[],
                                funcspec=[],
                                type=c_ast.Typename(
                                    name="int",
                                    type=c_ast.TypeDecl(
                                        declname="j",
                                        quals=[],
                                        type=c_ast.IdentifierType(["int"]),
                                    ),
                                ),
                                init=c_ast.Constant("int", "0"),
                                bitsize=None,
                            ),
                            cond=c_ast.BinaryOp(
                                "<",
                                c_ast.ID("j"),
                                c_ast.Constant("int", str(self.K)),
                            ),
                            next=c_ast.UnaryOp("p++", c_ast.ID("j")),
                            stmt=c_ast.Compound(
                                [
                                    # a_frag.x[i*K + j] = *((float*)((char*)ptr +
                                    # (i*stride + j)*sizeof(elem)))
                                    c_ast.Assignment(
                                        op="=",
                                        lvalue=c_ast.ArrayRef(
                                            name=c_ast.ID("a_frag.x"),
                                            subscript=c_ast.BinaryOp(
                                                "+",
                                                c_ast.BinaryOp(
                                                    "*",
                                                    c_ast.ID("i"),
                                                    c_ast.Constant(
                                                        "int", str(self.K)
                                                    ),
                                                ),
                                                c_ast.ID("j"),
                                            ),
                                        ),
                                        rvalue=c_ast.UnaryOp(
                                            op="*",
                                            expr=c_ast.Cast(
                                                to_type=c_ast.PtrDecl(
                                                    quals=[],
                                                    type=c_ast.TypeDecl(
                                                        declname=None,
                                                        quals=[],
                                                        type=c_ast.IdentifierType(
                                                            ["half"]
                                                        ),
                                                    ),
                                                ),
                                                expr=c_ast.UnaryOp(
                                                    op="&",
                                                    expr=c_ast.ID(
                                                        f"{ptr.name}[i*{stride.name}+j]"
                                                    ),
                                                ),
                                            ),
                                        ),
                                    )
                                ]
                            ),
                        )
                    ]
                ),
            )
            return loop_i

        # 2) mma_sync(ab_frag, a_frag, b_frag, ab_frag)
        if name.endswith("mma_sync"):
            # Expand 成  for p in 0..M*N-1: ab_frag.x[p] += a_frag.x[p] *
            # b_frag.x[p]
            idx = c_ast.Decl(
                name="p",
                quals=[],
                storage=[],
                funcspec=[],
                type=c_ast.Typename(
                    name="int",
                    type=c_ast.TypeDecl(
                        declname="p",
                        quals=[],
                        type=c_ast.IdentifierType(["int"]),
                    ),
                ),
                init=c_ast.Constant("int", "0"),
                bitsize=None,
            )
            loop_p = c_ast.For(
                init=idx,
                cond=c_ast.BinaryOp(
                    "<",
                    c_ast.ID("p"),
                    c_ast.Constant("int", str(self.M * self.N)),
                ),
                next=c_ast.UnaryOp("p++", c_ast.ID("p")),
                stmt=c_ast.Compound(
                    [
                        c_ast.Assignment(
                            op="+=",
                            lvalue=c_ast.ArrayRef(
                                c_ast.ID("ab_frag.x"), c_ast.ID("p")
                            ),
                            rvalue=c_ast.BinaryOp(
                                "*",
                                c_ast.ArrayRef(
                                    c_ast.ID("a_frag.x"), c_ast.ID("p")
                                ),
                                c_ast.ArrayRef(
                                    c_ast.ID("b_frag.x"), c_ast.ID("p")
                                ),
                            ),
                        )
                    ]
                ),
            )
            return loop_p

        # 3) store_matrix_sync(D + base, c_frag, stride, ...)
        if name.endswith("store_matrix_sync"):
            _, ptr, stride, _ = args
            # for i in 0..M-1: for j in 0..N-1:
            #   *((float*)((char*)ptr + (i*stride+j)*sizeof(elem))) = c_frag.x[i*N+j];
            loop_i = c_ast.For(
                init=c_ast.Decl(
                    name="i",
                    quals=[],
                    storage=[],
                    funcspec=[],
                    type=c_ast.Typename(
                        name="int",
                        type=c_ast.TypeDecl(
                            declname="i",
                            quals=[],
                            type=c_ast.IdentifierType(["int"]),
                        ),
                    ),
                    init=c_ast.Constant("int", "0"),
                    bitsize=None,
                ),
                cond=c_ast.BinaryOp(
                    "<", c_ast.ID("i"), c_ast.Constant("int", str(self.M))
                ),
                next=c_ast.UnaryOp("p++", c_ast.ID("i")),
                stmt=c_ast.Compound(
                    [
                        c_ast.For(
                            init=c_ast.Decl(
                                name="j",
                                quals=[],
                                storage=[],
                                funcspec=[],
                                type=c_ast.Typename(
                                    name="int",
                                    type=c_ast.TypeDecl(
                                        declname="j",
                                        quals=[],
                                        type=c_ast.IdentifierType(["int"]),
                                    ),
                                ),
                                init=c_ast.Constant("int", "0"),
                                bitsize=None,
                            ),
                            cond=c_ast.BinaryOp(
                                "<",
                                c_ast.ID("j"),
                                c_ast.Constant("int", str(self.N)),
                            ),
                            next=c_ast.UnaryOp("p++", c_ast.ID("j")),
                            stmt=c_ast.Compound(
                                [
                                    c_ast.Assignment(
                                        op="=",
                                        lvalue=c_ast.UnaryOp(
                                            "*",
                                            c_ast.Cast(
                                                to_type=c_ast.PtrDecl(
                                                    quals=[],
                                                    type=c_ast.TypeDecl(
                                                        declname=None,
                                                        quals=[],
                                                        type=c_ast.IdentifierType(
                                                            ["float"]
                                                        ),
                                                    ),
                                                ),
                                                expr=c_ast.BinaryOp(
                                                    "+",
                                                    ptr,
                                                    c_ast.BinaryOp(
                                                        "+",
                                                        c_ast.BinaryOp(
                                                            "*",
                                                            c_ast.ID("i"),
                                                            stride,
                                                        ),
                                                        c_ast.ID("j"),
                                                    ),
                                                ),
                                            ),
                                        ),
                                        rvalue=c_ast.ArrayRef(
                                            name=c_ast.ID("c_frag.x"),
                                            subscript=c_ast.BinaryOp(
                                                "+",
                                                c_ast.BinaryOp(
                                                    "*",
                                                    c_ast.ID("i"),
                                                    c_ast.Constant(
                                                        "int", str(self.N)
                                                    ),
                                                ),
                                                c_ast.ID("j"),
                                            ),
                                        ),
                                    )
                                ]
                            ),
                        )
                    ]
                ),
            )
            return loop_i

        # 其他保持原样
        return node


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


def preprocess(code: str) -> str:
    for pat, tmp in mapping.items():
        code = re.sub(pat, tmp, code)
    return code


def postprocess(code: str) -> str:
    # 还原：临时符号 → 原 CUDA 关键字
    for pat, tmp in mapping.items():
        # pat 是正则，这里直接写字面量
        orig = pat.replace(r"\b", "").replace("\\.", ".")
        code = code.replace(tmp, orig)
    return code


def ast_detensorization(code: str) -> str:
    # preprocess
    tmp = preprocess(code)
    print("[INFO]*****temp: ", tmp)
    parser = CParser()
    ast = parser.parse(tmp)
    transformer = TensorCoreToScalar(M=32, N=32, K=16)  # 根据实际改
    new_ast = transformer.visit(ast)
    generator = c_generator.CGenerator()
    new_tmp = generator.visit(new_ast)
    # postprocess
    postprocess(new_tmp)
    return new_tmp


if __name__ == "__main__":
    cuda_code = """
    __global__ void WMMAF16TensorCore(half* A, half* B, float* C, float* D)
    {
        int ix = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
        int iy = (blockIdx.y * blockDim.y + threadIdx.y);
        wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, M, N, K, float> ab_frag;
        wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;

        wmma::fill_fragment(ab_frag, 0.0f);
        int a_col, a_row, b_col, b_row, c_col, c_row;
        a_row = ix * M;
        b_row = iy * N;
        for (int k=0; k<32; k+=K) {
            a_col = b_col = k;

            if (a_row < 32 && a_col < 32 && b_row < 32 && b_col < 32) {
                // Load the inputs
                wmma::load_matrix_sync(a_frag, A + a_col + a_row * 32, 32);
                wmma::load_matrix_sync(b_frag, B + b_col + b_col * 32, 32);

                // Perform the matrix multiplication
                wmma::mma_sync(ab_frag, a_frag, b_frag, ab_frag);
            }
        }

        c_col = b_row;
        c_row = a_row;
        if (c_row < 32 && c_col < 32) {
            wmma::load_matrix_sync(c_frag, C + c_col + c_row * 32, 32, wmma::mem_row_major);

            for (int i = 0; i < c_frag.num_elements; i++) {
                c_frag.x[i] = ab_frag.x[i] + c_frag.x[i];
            }

            // Store the output
            wmma::store_matrix_sync(D + c_col + c_row * 32, c_frag, 32, wmma::mem_row_major);
        }
    }
    """
    scalar = ast_detensorization(cuda_code)
    print(scalar)
