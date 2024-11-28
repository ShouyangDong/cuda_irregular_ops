import json

from falcon.smt.auto_cache import ast_auto_cache
from falcon.smt.const_inline import constant_inline
from falcon.smt.tensorization.tensorization import ast_tensorization
from falcon.smt.thread_binding import ast_thread_binding
from falcon.src.post_processing.post_processing import (
    replace_operation_with_intrinsic,
    run_code_decoration,
)


def post_processing_pipeline(code, target):
    """This function transforms the given code by performing two main transformations:
        1. Convert parallel loop variables (e.g., OpenMP, CUDA) into standard C for loops.
        2. Convert SIMD tensor operations into scalar for-loop based calculations.
    :param func_content: The content of the function (code) to be transformed.

    :return: Transformed code after applying the two transformations."""
    code = constant_inline(code)
    code = ast_thread_binding(code, target)
    # when target is "BANG" or "DLBOOST", insert tensorization process.
    if target in ["BANG", "DLBOOST"]:
        code = run_code_decoration(code)
        print("[INFO] decorated: ", code)
        op_pragma = {}
        if target == "BANG":
            op_pragma = json.load(
                open("./falcon/documents/operation_bang_C_instruction_map.json", "r")
            )
        code, space_maps = replace_operation_with_intrinsic(code, op_pragma)
        print("[INFO] intrinsic: ", code)
        code = ast_auto_cache(code, space_maps)
        print("[INFO] cache: ", code)
        code = run_code_decoration(code)
        print("[INFO] decorate: ", code)
        code = ast_tensorization(code, target)
    return code


if __name__ == "__main__":
    code = """
    void add_kernel(float* output, float* input1, float* input2) {
        int dim1 = 4;
        int dim2 = 4;
        int dim3 = 4;
        int dim4 = 64;

        for (int i = 0; i < dim1; i++) {
            for (int j = 0; j < dim2; j++) {
                for (int k = 0; k < dim3; k++) {
                    for (int l = 0; l < dim4; l++) {
                        int index = i * dim2 * dim3 * dim4 + j * dim3 * dim4 + k * dim4 + l;
                        output[index] = input1[index] + input2[index];
                    }
                }
            }
        }
    }
    """
    code = post_processing_pipeline(code, "BANG")
    print(code)
