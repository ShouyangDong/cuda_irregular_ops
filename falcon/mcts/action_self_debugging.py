import json
import random

from falcon.smt.auto_cache import ast_auto_cache
from falcon.smt.loop_transformation.loop_contraction import \
    ast_loop_contraction
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
    run_apply_split, run_loop_contraction, run_loop_fusion, run_loop_reorder,
    run_split_annotation)
from falcon.src.post_processing.post_processing import (
    replace_operation_with_intrinsic, run_cache_process, run_code_decoration,
    run_double_buffer, run_tensorization, run_thread_binding)
from falcon.src.pre_processing.preprocessing import (run_detensorization,
                                                     run_loop_recovery)
from falcon.unit_test import unit_test

# Compile Script
compile_script = {
    "cpu": "benchmark/evaluation/dlboost_test/compilation.py",
    "MLU": "benchmark/evaluation/mlu_test/compilation.py",
    "cuda": "benchmark/evaluation/cuda_test/compilation.py"
    "hip": "benchmark/evaluation/hip_test/compilation.py"
}

# Test Script
test_script = {
    "cpu": "benchmark/evaluation/dlboost_test/result_test.py"
    "MLU": "benchmark/evaluation/mlu_test/result_test.py"
    "cuda": "benchmark/evaluation/cuda_test/result_test.py"
    "hip": "benchmark/evaluation/hip_test/result_test.py"
}


def fix_computation_code(source_code, error_code, error_output):
    prompt = f"""The error code is originally translated from given code with the same functionality.
    But the error code has some bugs and I cannot find them. Help me correct the error code. Return
    the fixed error code.

    source code:\n{source_code}\n
    error code:\n{error_code}\n
    error messag:\n{error_output}\n

    Please provide only the complete fixed C code based on error code without any additional text or explanations.
    Please make sure that you don't add any other text, just post back the code.
    It is very important that you do that, because otherwise you will interfere with a very important task of mine."""
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": question_system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    output = response["choices"][0]["message"]["content"]
    return re.search(r"```(?:cpp)?\s*(.*?)```", output, re.DOTALL)


def loop_recovery(file_name, code, source_platform, target_platform):
    final_code = run_loop_recovery(code, source_platform)
    success, output = unit_test(file_name, final_code)
    if success:
        return final_code

    for i in range(5):
        final_code = fix_computation_code(code, final_code, output)
        success, output = unit_test(file_name, final_code)
        if success:
            return final_code

    return final_code


def stmt_split(file_name, code, source_platform, target_platform):
    # TODO:add llm stmt split and unit test
    return ast_stmt_split(code)


def detensorization(file_name, code, source_platform, target_platform):
    final_code = run_detensorization(code, source_platform)
    success, output = unit_test(file_name, final_code)
    if success:
        return final_code

    for i in range(5):
        final_code = fix_computation_code(code, final_code, output)
        success, output = unit_test(file_name, final_code)
        if success:
            return final_code
    return final_code


def loop_fusion(file_name, code, source_platform, target_platform):
    final_code = run_loop_fusion(code)
    success, output = unit_test(file_name, final_code)
    if success:
        return final_code

    for i in range(5):
        final_code = fix_computation_code(code, final_code, output)
        success, output = unit_test(file_name, final_code)
        if success:
            return final_code
    return final_code


def loop_reorder(file_name, code, source_platform, target_platform):
    final_code = run_loop_reorder(code)
    success, output = unit_test(file_name, final_code)
    if success:
        return final_code

    for i in range(5):
        final_code = fix_computation_code(code, final_code, output)
        success, output = unit_test(file_name, final_code)
        if success:
            return final_code
    return final_code


def loop_split(file_name, code, source_platform, target_platform):
    code = run_split_annotation(code)
    success, output = unit_test(file_name, final_code)
    if success:
        return final_code

    for i in range(5):
        final_code = fix_computation_code(code, final_code, output)
        success, output = unit_test(file_name, final_code)
        if success:
            return final_code
    return final_code


def loop_contraction(file_name, code, source_platform, target_platform):
    final_code = run_loop_contraction(code)
    success, output = unit_test(file_name, final_code)
    if success:
        return final_code

    for i in range(5):
        final_code = fix_computation_code(code, final_code, output)
        success, output = unit_test(file_name, final_code)
        if success:
            return final_code
    return final_code


def auto_bind(file_name, code, source_platform, target_platform):
    if target_platform not in ["mlu", "cuda", "hip"]:
        return code

    final_code = run_thread_binding(code, target_platform)
    success, output = unit_test(file_name, final_code)
    if success:
        return final_code

    for i in range(5):
        final_code = fix_computation_code(code, final_code, output)
        success, output = unit_test(file_name, final_code)
        if success:
            return final_code
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

    cache_code = run_cache_process(code, space_maps)
    success, output = unit_test(file_name, final_code)
    if success:
        return final_code

    for i in range(5):
        final_code = fix_computation_code(code, final_code, output)
        success, output = unit_test(file_name, final_code)
        if success:
            return final_code
    return cache_code


def auto_tensorization(file_name, code, source_platform, target_platform):
    code = run_code_decoration(code)
    final_code = run_tensorization(code, target_platform)
    success, output = unit_test(file_name, final_code)
    if success:
        return final_code

    for i in range(5):
        final_code = fix_computation_code(code, final_code, output)
        success, output = unit_test(file_name, final_code)
        if success:
            return final_code
    return final_code


def auto_pipeline(file_name, code, source_platform, target_platform):
    if target_platform not in ["mlu"]:
        return code

    final_code = run_double_buffer(code, target_platform)
    success, output = unit_test(file_name, final_code)
    if success:
        return final_code

    for i in range(5):
        final_code = fix_computation_code(code, final_code, output)
        success, output = unit_test(file_name, final_code)
        if success:
            return final_code
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
