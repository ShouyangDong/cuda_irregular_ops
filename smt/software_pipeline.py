from string import Template

BANG_binary_template = Template(
    """__mlu_entry__ void binary_double_buffering(float* OUTPUT， float* INPUT0, float* INPUT1, int BUF_SIZE, int loop_ext) {
    __nram__ float INPUT0_N[BUF_SIZE * 2];
    __nram__ float INPUT1_N[BUF_SIZE * 2];
    __nram__ float OUTPUT_N[BUF_SIZE * 2];

    __memcpy_async(INPUT0_N, INPUT0, BUF_SIZE * sizeof(float), GDRAM2NRAM);
    __asm__ volatile("sync;");
    __memcpy_async(INPUT1_N, INPUT1, BUF_SIZE * sizeof(float), GDRAM2NRAM);
    __asm__ volatile("sync;");
    __memcpy_async(INPUT0_N + BUF_SIZE, INPUT0 + BUF_SIZE, BUF_SIZE * sizeof(float), GDRAM2NRAM);
    __memcpy_async(INPUT1_N + BUF_SIZE, INPUT1 + BUF_SIZE, BUF_SIZE * sizeof(float), GDRAM2NRAM);
    {inst}(OUTPUT_N, INPUT0_N, INPUT1_N, BUF_SIZE);
    __asm__ volatile("sync;");
    for (int i_outer = 0; i_outer < (loop_ext / 2 - 1); ++i_outer) {
        __memcpy_async(INPUT0_N, INPUT0 + ((i_outer + 1) * BUF_SIZE * 2), BUF_SIZE * sizeof(float), GDRAM2NRAM);
        __memcpy_async(INPUT1_N, INPUT1 + ((i_outer + 1) * BUF_SIZE * 2), BUF_SIZE * sizeof(float), GDRAM2NRAM);
        {inst}(OUTPUT_N + BUF_SIZE, INPUT0_N + BUF_SIZE, INPUT1_N + BUF_SIZE, BUF_SIZE);
        __memcpy_async(OUTPUT + (i_outer * BUF_SIZE * 2) , OUTPUT_N, BUF_SIZE * sizeof(float), NRAM2GDRAM);   
        __asm__ volatile("sync;");
        __memcpy_async(INPUT0_N + BUF_SIZE, INPUT0 + ((i_outer + 1) * BUF_SIZE * 2) + BUF_SIZE, BUF_SIZE * sizeof(float), GDRAM2NRAM);
        __memcpy_async(INPUT1_N + BUF_SIZE, INPUT1 + ((i_outer + 1) * BUF_SIZE * 2) + BUF_SIZE, BUF_SIZE * sizeof(float), GDRAM2NRAM);
        __memcpy_async(OUTPUT + ((i_outer * BUF_SIZE * 2) + BUF_SIZE), OUTPUT_N + BUF_SIZE, BUF_SIZE * sizeof(float), NRAM2GDRAM);
        __asm__ volatile("sync;");
    }

    {inst}(OUTPUT_N + BUF_SIZE, INPUT0_N + BUF_SIZE, INPUT1_N + BUF_SIZE, BUF_SIZE);
    __memcpy_async(OUTPUT + (BUF_SIZE * loop_ext - BUF_SIZE * 2), OUTPUT_N, BUF_SIZE * sizeof(float), NRAM2GDRAM);
    __asm__ volatile("sync;");
    {inst}(OUTPUT_N + BUF_SIZE, INPUT0_N + BUF_SIZE, INPUT1_N + BUF_SIZE, BUF_SIZE);
    __memcpy_async(OUTPUT + (BUF_SIZE * loop_ext - BUF_SIZE), OUTPUT_N + BUF_SIZE, BUF_SIZE * sizeof(float), NRAM2GDRAM);
    __asm__ volatile("sync;");
}"""
)

BANG_unary_template = Template(
    """__mlu_entry__ void unary_double_buffering(float* OUTPUT， float* INPUT0, int BUF_SIZE, int loop_ext) {
    __nram__ float INPUT0_N[BUF_SIZE * 2];
    __nram__ float OUTPUT_N[BUF_SIZE * 2];

    __memcpy_async(INPUT0_N, INPUT0, BUF_SIZE * sizeof(float), GDRAM2NRAM);
    __asm__ volatile("sync;");
    __memcpy_async(INPUT0_N + BUF_SIZE, INPUT0 + BUF_SIZE, BUF_SIZE * sizeof(float), GDRAM2NRAM);
    {inst}(OUTPUT_N, INPUT0_N, BUF_SIZE);
    __asm__ volatile("sync;");
    for (int i_outer = 0; i_outer < (loop_ext / 2 - 1); ++i_outer) {
        __memcpy_async(INPUT0_N, INPUT0 + ((i_outer + 1) * BUF_SIZE * 2), BUF_SIZE * sizeof(float), GDRAM2NRAM);
        {inst}(OUTPUT_N + BUF_SIZE, INPUT0_N + BUF_SIZE, BUF_SIZE);
        __memcpy_async(OUTPUT + (i_outer * BUF_SIZE * 2) , OUTPUT_N, BUF_SIZE * sizeof(float), NRAM2GDRAM);   
        __asm__ volatile("sync;");
        __memcpy_async(INPUT0_N + BUF_SIZE, INPUT0 + ((i_outer + 1) * BUF_SIZE * 2) + BUF_SIZE, BUF_SIZE * sizeof(float), GDRAM2NRAM);
        __memcpy_async(OUTPUT + ((i_outer * BUF_SIZE * 2) + BUF_SIZE), OUTPUT_N + BUF_SIZE, BUF_SIZE * sizeof(float), NRAM2GDRAM);
        __asm__ volatile("sync;");
    }
    
    {inst}(OUTPUT_N + BUF_SIZE, INPUT0_N + BUF_SIZE, BUF_SIZE);
    __memcpy_async(OUTPUT + (BUF_SIZE * loop_ext - BUF_SIZE * 2), OUTPUT_N, BUF_SIZE * sizeof(float), NRAM2GDRAM);
    __asm__ volatile("sync;");
    {inst}(OUTPUT_N + BUF_SIZE, INPUT0_N + BUF_SIZE,  BUF_SIZE);
    __memcpy_async(OUTPUT + (BUF_SIZE * loop_ext - BUF_SIZE), OUTPUT_N + BUF_SIZE, BUF_SIZE * sizeof(float), NRAM2GDRAM);
    __asm__ volatile("sync;");
}"""
)


from pycparser import c_ast, c_generator, c_parser

from smt.util import NodeTransformer


class PragmaVisitor(NodeTransformer):
    def __init__(self):
        self.pragma_info = {}

    def visit_Compound(self, node):
        # Get the block_items
        blocks = node.block_items
        for index, node in enumerate(blocks):
            if isinstance(node, c_ast.Pragma) and node.string == "software_pipeline":
                # self.pragma_info[node.string] = blocks[index + 1]
                pipeline_block = blocks[index + 1]
                assert isinstance(pipeline_block, c_ast.For)
                ext = pipeline_block.cond.right.value

                new_call = None
                for stmt in pipeline_block.stmt.block_items:
                    if "__bang" in stmt.name.name:
                        ext_const = c_ast.Constant(type="int", value=ext)

                        new_args = stmt.args.exprs + [ext_const]
                        new_call = c_ast.FuncCall(
                            name=c_ast.ID(name="binary_double_buffering"), args=new_args
                        )
                return new_call
        return new_call


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
    parser = c_parser.CParser()
    ast = parser.parse(code)
    visitor = PragmaVisitor()
    visitor.visit(ast)
    print("ast: ", ast)
    generator = c_generator.CGenerator()
    print(generator.visit(ast))
