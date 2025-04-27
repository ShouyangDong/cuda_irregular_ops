import json

from pycparser import c_ast, c_generator, c_parser

from falcon.buffer_inline import ast_buffer_inline
from falcon.simplification import simplify_code
from falcon.stmt_simplification import ast_stmt_simplification
from falcon.util import NodeTransformer, make_full_func, remove_target_prefix

mlu_file_name = "falcon/documents/mlu_op_tensorization.json"
cuda_file_name = "falcon/documents/cuda_op_tensorization.json"
hip_file_name = "falcon/documents/hip_op_tensorization.json"
cpu_file_name = "falcon/documents/cpu_op_tensorization.json"


class Detensorizer(NodeTransformer):
    def __init__(self, func_defs):
        self.func_defs = func_defs
        self.parser = c_parser.CParser()
        self.parameter_mappings = {}

    def visit_FuncCall(self, node):
        if node.name.name in self.func_defs:
            func_def = self.func_defs[node.name.name]
            seq_def = self.parser.parse(func_def)
            if not isinstance(seq_def, c_ast.FileAST):
                raise ValueError("Sequential code must be a function")

            # Construct a map between the function call's  arguments and
            # callee's arguments
            seq_def_args = seq_def.ext[0].decl.type.args.params
            seq_def_name = [arg_id.name for arg_id in seq_def_args]
            self.parameter_mappings = {
                arg: param for arg, param in zip(seq_def_name, node.args.exprs)
            }
            body = seq_def.ext[0].body
            return self.visit(body.block_items[0])
        else:
            return self.generic_visit(node)

    def visit_ID(self, node):
        if node.name in self.parameter_mappings:
            return self.parameter_mappings[node.name]
        return node


def ast_detensorization(code, target):
    """
    Transform C code using an SMT solver to optimize loop constructs.

    This function parses the provided C code into an Abstract Syntax Tree (AST) and applies
    a transformation to split loops based on the given loop index and factor. The transformation
    is guided by an SMT solver to ensure the generated code is logically equivalent to the
    original but potentially more optimized.

    Parameters:
    - code (str): A string containing the C code to be transformed.
    - file_name (str): The definition of intrinsics.

    Returns:
    - str: The transformed C code as a string.

    Todo:
    - Implement additional error checking for the input parameters.
    - Extend the visitor to handle more complex loop structures.
    """
    code = remove_target_prefix(code)
    if target == "mlu":
        parser = c_parser.CParser()
        ast = parser.parse(code)
        with open(mlu_file_name) as json_file:
            func_defs = json.load(json_file)
        visitor = Detensorizer(func_defs)
        visitor.visit(ast)
        generator = c_generator.CGenerator()
        code = generator.visit(ast)

    elif target == "cuda":
        parser = c_parser.CParser()
        ast = parser.parse(code)
        with open(cuda_file_name) as json_file:
            func_defs = json.load(json_file)
        visitor = Detensorizer(func_defs)
        visitor.visit(ast)
        generator = c_generator.CGenerator()
        code = generator.visit(ast)

    elif target == "hip":
        parser = c_parser.CParser()
        ast = parser.parse(code)
        with open(hip_file_name) as json_file:
            func_defs = json.load(json_file)
        visitor = Detensorizer(func_defs)
        visitor.visit(ast)
        generator = c_generator.CGenerator()
        code = generator.visit(ast)

    elif target == "cpu":
        parser = c_parser.CParser()
        ast = parser.parse(code)
        with open(cpu_file_name) as json_file:
            func_defs = json.load(json_file)
        visitor = Detensorizer(func_defs)
        visitor.visit(ast)
        generator = c_generator.CGenerator()
        code = generator.visit(ast)

    else:
        raise RuntimeError("unsuppored target.")
    code = simplify_code(code)
    code = ast_stmt_simplification(code)
    code = ast_buffer_inline(code)
    code = make_full_func(code, target)
    return code


if __name__ == "__main__":
    code = """
    void add(float* lhs, float* rhs, float* add_1515) {
        float lhs_local_nram[128];
        __memcpy(((float *)lhs_local_nram + (0)), ((float *)lhs + (((((int)clusterId) * 256) + (((int)coreId) * 64)))), 256, GDRAM2NRAM);
        __memcpy(((float *)lhs_local_nram + (64)), ((float *)rhs + (((((int)clusterId) * 256) + (((int)coreId) * 64)))), 256, GDRAM2NRAM);
        __bang_add(((float *)lhs_local_nram + (0)), ((float *)lhs_local_nram + (0)), ((float *)lhs_local_nram + (64)), 64);
        __memcpy(((float *)add_1515 + (((((int)clusterId) * 256) + (((int)coreId) * 64)))), ((float *)lhs_local_nram + (0)), 256, NRAM2GDRAM);
    }
    """
    code = ast_detensorization(code, "mlu")
    print(code)
    code = """
        void tanh(float* input0, float* active_tanh_210) {
        float input0_local_nram[640];
        __memcpy(((float *)input0_local_nram + (0)), ((float *)input0 + (((((int)clusterId) * 2560) + (((int)coreId) * 640)))), 2560, GDRAM2NRAM);
        __bang_active_tanh(((float *)input0_local_nram + (0)), ((float *)input0_local_nram + (0)), 640);
        __memcpy(((float *)active_tanh_210 + (((((int)clusterId) * 2560) + (((int)coreId) * 640)))), ((float *)input0_local_nram + (0)), 2560, NRAM2GDRAM);
    }
    """
    code = ast_detensorization(code, "mlu")
    print(code)
    code = """
    void softmax(float *A, float *output)
    {
        for (int clusterId = 0; clusterId < 4; ++clusterId)
        {
            for (int coreId = 0; coreId < 4; ++coreId)
            {
                float dest[128];
                float dinominator[128];
                float dinominator_temp[128];
                float src1[128];
                float addition[128];
                for (int i = (clusterId * 4) + coreId; i < 5; i += 16)
                {
                    __memcpy(src1, A + (i * 128), 512, GDRAM2NRAM);
                    __bang_active_exp(src1, src1, 128);
                    __bang_write_zero(dinominator, 128);
                    __bang_sumpool(dinominator, src1, 1, 1, 128, 1, 128, 1, 1);
                    __memset_nram(dinominator_temp, 128, dinominator[0]);
                    __bang_div(dest, src1, dinominator_temp, addition, 128);
                    __memcpy(output + (128 * i), dest, 512, NRAM2GDRAM);
                }
            }
        }
    }
    """
    code = ast_detensorization(code, "mlu")
    print(code)

    bang_code = """
    extern "C" __mlu_global__ void gemm(float *A, float *B, float *C) {
        __nram__ float A_nram[8 * 128];
        __wram__ float B_wram[128 * 128];
        __nram__ float C_nram[8 * 128];
        if (clusterId < 4) {
            if (coreId < 4) {
            __memcpy(A_nram, A + (clusterId * 4 + coreId) * 8 * 128, 8 * 128 * 4,
                    GDRAM2NRAM);
            __memcpy(B_wram, B, 128 * 128 * 4, GDRAM2WRAM);

            __bang_matmul(C_nram, A_nram, B_wram, 8, 128, 128);
            __memcpy(C + (clusterId * 4 + coreId) * 8 * 128, C_nram, 8 * 128 * 4,
                    NRAM2GDRAM);
            }
        }
    }
    """
    converted_code = ast_detensorization(bang_code, "mlu")
    print(converted_code)

    bang_code = """
    extern "C" __mlu_global__ void add(float *lhs, float *rhs, float *add_1605) {
    __nram__ float lhs_local_nram[512];
    if (((((int)clusterId) * 4) + ((int)coreId)) < 9) {
        __memcpy(
            ((float *)lhs_local_nram + (0)),
            ((float *)lhs + (((((int)clusterId) * 1024) + (((int)coreId) * 256)))),
            1024, GDRAM2NRAM);
    }
    if (((((int)clusterId) * 4) + ((int)coreId)) < 9) {
        __memcpy(
            ((float *)lhs_local_nram + (256)),
            ((float *)rhs + (((((int)clusterId) * 1024) + (((int)coreId) * 256)))),
            1024, GDRAM2NRAM);
    }
    if (((((int)clusterId) * 4) + ((int)coreId)) < 9) {
        __bang_add(((float *)lhs_local_nram + (0)), ((float *)lhs_local_nram + (0)),
                ((float *)lhs_local_nram + (256)), 256);
    }
    if (((((int)clusterId) * 4) + ((int)coreId)) < 9) {
        __memcpy(((float *)add_1605 +
                (((((int)clusterId) * 1024) + (((int)coreId) * 256)))),
                ((float *)lhs_local_nram + (0)), 1024, NRAM2GDRAM);
    }
    }
    """
    converted_code = ast_detensorization(bang_code, "mlu")
    print(converted_code)
