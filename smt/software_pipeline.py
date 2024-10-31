from string import Template

BANG_binary_template = Template(
    """void binary_double_buffering(float* OUTPUT， float* INPUT0, float* INPUT1, int BUF_SIZE, int loop_ext) {
    __nram__ float INPUT0_N[BUF_SIZE * 2];
    __nram__ float INPUT1_N[BUF_SIZE * 2];
    __nram__ float OUTPUT_N[BUF_SIZE * 2];

    __memcpy_async(INPUT0_N, INPUT0, BUF_SIZE * sizeof(float), GDRAM2NRAM);
    __asm__ volatile("sync;");
    __memcpy_async(INPUT1_N, INPUT1, BUF_SIZE * sizeof(float), GDRAM2NRAM);
    __asm__ volatile("sync;");
    __memcpy_async(INPUT0_N + BUF_SIZE, INPUT0 + BUF_SIZE, BUF_SIZE * sizeof(float), GDRAM2NRAM);
    __memcpy_async(INPUT1_N + BUF_SIZE, INPUT1 + BUF_SIZE, BUF_SIZE * sizeof(float), GDRAM2NRAM);
    $inst(OUTPUT_N, INPUT0_N, INPUT1_N, BUF_SIZE);
    __asm__ volatile("sync;");
    for (int i_outer = 0; i_outer < (loop_ext / 2 - 1); ++i_outer) {
        __memcpy_async(INPUT0_N, INPUT0 + ((i_outer + 1) * BUF_SIZE * 2), BUF_SIZE * sizeof(float), GDRAM2NRAM);
        __memcpy_async(INPUT1_N, INPUT1 + ((i_outer + 1) * BUF_SIZE * 2), BUF_SIZE * sizeof(float), GDRAM2NRAM);
        $inst(OUTPUT_N + BUF_SIZE, INPUT0_N + BUF_SIZE, INPUT1_N + BUF_SIZE, BUF_SIZE);
        __memcpy_async(OUTPUT + (i_outer * BUF_SIZE * 2) , OUTPUT_N, BUF_SIZE * sizeof(float), NRAM2GDRAM);   
        __asm__ volatile("sync;");
        __memcpy_async(INPUT0_N + BUF_SIZE, INPUT0 + ((i_outer + 1) * BUF_SIZE * 2) + BUF_SIZE, BUF_SIZE * sizeof(float), GDRAM2NRAM);
        __memcpy_async(INPUT1_N + BUF_SIZE, INPUT1 + ((i_outer + 1) * BUF_SIZE * 2) + BUF_SIZE, BUF_SIZE * sizeof(float), GDRAM2NRAM);
        __memcpy_async(OUTPUT + ((i_outer * BUF_SIZE * 2) + BUF_SIZE), OUTPUT_N + BUF_SIZE, BUF_SIZE * sizeof(float), NRAM2GDRAM);
        __asm__ volatile("sync;");
    }

    $inst(OUTPUT_N + BUF_SIZE, INPUT0_N + BUF_SIZE, INPUT1_N + BUF_SIZE, BUF_SIZE);
    __memcpy_async(OUTPUT + (BUF_SIZE * loop_ext - BUF_SIZE * 2), OUTPUT_N, BUF_SIZE * sizeof(float), NRAM2GDRAM);
    __asm__ volatile("sync;");
    $inst(OUTPUT_N + BUF_SIZE, INPUT0_N + BUF_SIZE, INPUT1_N + BUF_SIZE, BUF_SIZE);
    __memcpy_async(OUTPUT + (BUF_SIZE * loop_ext - BUF_SIZE), OUTPUT_N + BUF_SIZE, BUF_SIZE * sizeof(float), NRAM2GDRAM);
    __asm__ volatile("sync;");
}\n"""
)

BANG_unary_template = Template(
    """void unary_double_buffering(float* OUTPUT， float* INPUT0, int BUF_SIZE, int loop_ext) {
    __nram__ float INPUT0_N[BUF_SIZE * 2];
    __nram__ float OUTPUT_N[BUF_SIZE * 2];

    __memcpy_async(INPUT0_N, INPUT0, BUF_SIZE * sizeof(float), GDRAM2NRAM);
    __asm__ volatile("sync;");
    __memcpy_async(INPUT0_N + BUF_SIZE, INPUT0 + BUF_SIZE, BUF_SIZE * sizeof(float), GDRAM2NRAM);
    $inst(OUTPUT_N, INPUT0_N, BUF_SIZE);
    __asm__ volatile("sync;");
    for (int i_outer = 0; i_outer < (loop_ext / 2 - 1); ++i_outer) {
        __memcpy_async(INPUT0_N, INPUT0 + ((i_outer + 1) * BUF_SIZE * 2), BUF_SIZE * sizeof(float), GDRAM2NRAM);
        $inst(OUTPUT_N + BUF_SIZE, INPUT0_N + BUF_SIZE, BUF_SIZE);
        __memcpy_async(OUTPUT + (i_outer * BUF_SIZE * 2) , OUTPUT_N, BUF_SIZE * sizeof(float), NRAM2GDRAM);   
        __asm__ volatile("sync;");
        __memcpy_async(INPUT0_N + BUF_SIZE, INPUT0 + ((i_outer + 1) * BUF_SIZE * 2) + BUF_SIZE, BUF_SIZE * sizeof(float), GDRAM2NRAM);
        __memcpy_async(OUTPUT + ((i_outer * BUF_SIZE * 2) + BUF_SIZE), OUTPUT_N + BUF_SIZE, BUF_SIZE * sizeof(float), NRAM2GDRAM);
        __asm__ volatile("sync;");
    }
    
    $inst(OUTPUT_N + BUF_SIZE, INPUT0_N + BUF_SIZE, BUF_SIZE);
    __memcpy_async(OUTPUT + (BUF_SIZE * loop_ext - BUF_SIZE * 2), OUTPUT_N, BUF_SIZE * sizeof(float), NRAM2GDRAM);
    __asm__ volatile("sync;");
    $inst(OUTPUT_N + BUF_SIZE, INPUT0_N + BUF_SIZE,  BUF_SIZE);
    __memcpy_async(OUTPUT + (BUF_SIZE * loop_ext - BUF_SIZE), OUTPUT_N + BUF_SIZE, BUF_SIZE * sizeof(float), NRAM2GDRAM);
    __asm__ volatile("sync;");
}\n"""
)

op_map = {
    "__bang_add": "binary_double_buffering",
    "__bang_active_tanh": "unary_double_buffering",
}
op_template = {
    "__bang_add": BANG_binary_template,
    "__bang_active_tanh": BANG_unary_template,
}
from pycparser import c_ast, c_generator, c_parser

from smt.util import NodeTransformer


class PragmaVisitor(NodeTransformer):
    def __init__(self):
        self.inst = []

    def visit_Compound(self, node):
        # 获取 `block_items`
        blocks = node.block_items
        if not blocks:
            return node

        new_block_items = []
        skip_next = False

        # 遍历 `block_items`，查找 `Pragma` 和 `for` 组合
        for index, subnode in enumerate(blocks):
            if skip_next:
                # 跳过下一个节点（for 循环），因为已经处理过
                skip_next = False
                continue

            # 检查是否是 `#pragma software_pipeline`
            if (
                isinstance(subnode, c_ast.Pragma)
                and subnode.string == "software_pipeline"
            ):
                if index + 1 < len(blocks) and isinstance(blocks[index + 1], c_ast.For):
                    pipeline_for = blocks[index + 1]

                    ext = pipeline_for.cond.right.value

                    new_call = None
                    for stmt in pipeline_for.stmt.block_items:
                        if "__bang" in stmt.name.name:
                            self.inst = stmt.name.name
                            ext_const = c_ast.Constant(type="int", value=ext)
                            new_args = stmt.args.exprs + [ext_const]
                            new_call = c_ast.FuncCall(
                                name=c_ast.ID(name=op_map[self.inst]),
                                args=c_ast.ExprList(new_args),
                            )

                    # 添加替换后的调用
                    new_block_items.append(new_call)

                    # 设置跳过下一个 `for` 循环
                    skip_next = True
                else:
                    # 如果没有找到 `for`，继续添加当前节点
                    new_block_items.append(subnode)
            else:
                # 如果不是 `#pragma` 或 `for`，直接添加节点
                new_block_items.append(subnode)

        # 替换 `block_items`
        node.block_items = new_block_items
        return node


def smt_double_buffer(code):
    parser = c_parser.CParser()
    ast = parser.parse(code)
    visitor = PragmaVisitor()
    visitor.visit(ast)
    output_code = op_template[visitor.inst].substitute(inst=visitor.inst)
    generator = c_generator.CGenerator()
    modify_code = generator.visit(ast)
    return output_code + modify_code


if __name__ == "__main__":
    code = """void add(float* INPUT0, float* INPUT1, float* OUTPUT) {
        float INPUT0_N[64];
        float INPUT1_N[64];
        float OUTPUT_N[64];
        #pragma software_pipeline
        for (int i = 0; i < 2048; ++i) {
            __memcpy(INPUT0_N, INPUT0 + (i * 64), 256, GDRAM2NRAM);
            __memcpy(INPUT1_N, INPUT1 + (i * 64), 256, GDRAM2NRAM);
            __bang_add(OUTPUT_N, INPUT0_N , INPUT1_N, 64);
            __memcpy(OUTPUT + (i * 64), OUTPUT_N, 256, NRAM2GDRAM);
        }
    }
    """
    code = smt_double_buffer(code)
    print(code)
