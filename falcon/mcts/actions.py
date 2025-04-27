import json
import random

from falcon.smt.auto_cache import ast_auto_cache
from falcon.smt.loop_transformation.loop_contraction import (
    ast_loop_contraction,
)
from falcon.smt.loop_transformation.loop_fusion import ast_loop_fusion
from falcon.smt.loop_transformation.loop_recovery import ast_loop_recovery
from falcon.smt.loop_transformation.loop_reorder import ast_loop_reorder
from falcon.smt.loop_transformation.loop_split import ast_loop_split
from falcon.smt.software_pipeline import smt_double_buffer
from falcon.smt.stmt_split import ast_stmt_split
from falcon.smt.tensorization.detensorization import ast_detensorization
from falcon.smt.tensorization.tensorization import ast_tensorization
from falcon.smt.thread_binding import ast_thread_binding
from falcon.src.loop_transformation.loop_transformation import (
    run_apply_split,
    run_loop_contraction,
    run_loop_fusion,
    run_loop_reorder,
    run_split_annotation,
)
from falcon.src.post_processing.post_processing import (
    replace_operation_with_intrinsic,
    run_cache_process,
    run_code_decoration,
    run_double_buffer,
    run_tensorization,
    run_thread_binding,
)
from falcon.src.pre_processing.preprocessing import (
    run_detensorization,
    run_loop_recovery,
)
from falcon.unit_test import unit_test


def loop_recovery(file_name, code, source_platform, target_platform):
    try:
        final_code = run_loop_recovery(code, source_platform)
        if not unit_test(file_name, final_code):
            raise RuntimeError("loop recovery error")
    except RuntimeError:
        final_code = ast_loop_recovery(code, source_platform)
    return final_code


def stmt_split(file_name, code, source_platform, target_platform):
    # TODO:add llm stmt split and unit test
    return ast_stmt_split(code, target_platform)


def detensorization(file_name, code, source_platform, target_platform):
    try:
        final_code = run_detensorization(code, source_platform)
        if not unit_test(file_name, final_code):
            raise RuntimeError("detensorization error")
    except RuntimeError:
        final_code = ast_detensorization(code, source_platform)
    return final_code


def loop_fusion(file_name, code, source_platform, target_platform):
    try:
        final_code = run_loop_fusion(code)
        if not unit_test(file_name, final_code):
            raise RuntimeError("loop fusion error")
    except RuntimeError as e:
        print("[INFO]loop fusion error: ", e)
        final_code = ast_loop_fusion(code)
    return final_code


def loop_reorder(file_name, code, source_platform, target_platform):
    print("[INFO]***********loop reorder code: ", code)
    try:
        final_code = run_loop_reorder(code)
        if not unit_test(file_name, final_code):
            raise RuntimeError("loop_reorder error")
    except RuntimeError as e:
        print("[INFO]loop reorder error: ", e)
        final_code = ast_loop_reorder(code)
    print("[INFO]***************final code: ", final_code)
    return final_code


def loop_split(file_name, code, source_platform, target_platform):
    try:
        code = run_split_annotation(code)
        final_code = run_apply_split(code)
        if not unit_test(file_name, final_code):
            raise RuntimeError("loop_split error")
    except RuntimeError as e:
        print("[INFO]loop split error: ", e)
        final_code = ast_loop_split(code)
    return final_code


def loop_contraction(file_name, code, source_platform, target_platform):
    try:
        final_code = run_loop_contraction(code, None)
        if not unit_test(file_name, final_code):
            raise RuntimeError("loop_contraction error")
    except RuntimeError:
        final_code = ast_loop_contraction(code)
    return final_code


def auto_bind(file_name, code, source_platform, target_platform):
    if target_platform not in ["mlu", "cuda", "hip"]:
        return code
    try:
        final_code = run_thread_binding(code, target_platform)
        if not unit_test(file_name, final_code):
            raise RuntimeError("auto_bind error")
    except RuntimeError:
        final_code = ast_thread_binding(code, target_platform)
    return final_code


def auto_cache(file_name, code, source_platform, target_platform):
    code = run_code_decoration(code)
    op_pragma = {}
    if target_platform == "mlu":
        op_pragma = json.load(
            open(
                "./falcon/documents/operation_bang_C_instruction_map.json", "r"
            )
        )
    code, space_maps = replace_operation_with_intrinsic(code, op_pragma)
    # If no need to cache, just return origin code
    if space_maps is None:
        return code
    try:
        cache_code = run_cache_process(code, space_maps)

        if not unit_test(file_name, cache_code):
            raise RuntimeError("auto_cache error")
    except RuntimeError:
        cache_code = ast_auto_cache(code, space_maps)
    return cache_code


def auto_tensorization(file_name, code, source_platform, target_platform):
    try:
        code = run_code_decoration(code)
        final_code = run_tensorization(code, target_platform)
        if not unit_test(file_name, final_code):
            raise RuntimeError("auto_tensorization error")
    except RuntimeError:
        final_code = ast_tensorization(code, target_platform)
    return final_code


def auto_pipeline(file_name, code, source_platform, target_platform):
    if target_platform not in ["mlu"]:
        return code
    try:
        final_code = run_double_buffer(code, target_platform)
        if not unit_test(file_name, final_code):
            raise RuntimeError("auto_pipeline error")
    except RuntimeError:
        final_code = smt_double_buffer(code)
    return final_code


actions = [
    loop_recovery,
    stmt_split,
    detensorization,
    loop_fusion,
    loop_reorder,
    loop_split,
    loop_contraction,
    auto_bind,
    auto_cache,
    auto_tensorization,
    auto_pipeline,
]

if __name__ == "__main__":
    code = """
    __global__ void __launch_bounds__(1024)
    add(float *__restrict__ A, float *__restrict__ B,
        float *__restrict__ T_add) {
        if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 2309) {
            T_add[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] =
                (A[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] +
                B[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
        }
    }
    """
    source_platform = "cuda"
    target_platform = "mlu"
    file_name = "benchmark/data/cuda_code_test/add_18_128.cu"
    selected_function = random.choice(actions)
    # 调用随机选择的函数
    result = selected_function(
        file_name, code, source_platform, target_platform
    )

    # 输出结果
    print("运行结果:", result)
