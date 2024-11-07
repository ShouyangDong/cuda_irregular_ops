from smt.loop_transformation.loop_recovery import ast_loop_recovery
from smt.tensorization.detensorization import ast_detensorization
from falcon.src.loop_transformation.loop_transformation import (
    run_apply_split,
    run_loop_fusion,
    run_split_annotation,
)
from falcon.src.pre_processing.preprocessing import run_detensorization, run_loop_recovery
from falcon.unit_test import unit_test

def run_transcompile_code(code, source, target):
    device_code = code.split("extern")[0]
    host_code = "extern" + code.split("extern")[1]

    code = run_loop_recovery(device_code, target)
    if not unit_test(file_name, device_code + host_code, target):
        code = ast_loop_recovery(device_code, target)

    if target in ["BANG"]:
        modi_code = run_detensorization(code, target)
        if not unit_test(file_name, modi_code + host_code, target):
            modi_code = ast_detensorization(code, target)

    fusion_code = run_loop_fusion(modi_code)
    if not unit_test(file_name, fusion_code, target):
        fusion_code = ast_loop_fusion(modi_code)

    code = run_split_annotation(fusion_code)

    split_code = run_apply_split(code)
    if not unit_test(file_name, split_code, target):
        split_code = ast_apply_split(code)

    final_code = run_thread_binding(split_code, target)
    if not unit_test(file_name, final_code + host_code, target):
        final_code = ast_thread_binding(split_code, target)

    return final_code


if __name__ == "__main__":
    file_path = "benchmark/data/mlu_code_test/add_4_4_4_64.mlu"
    with open(file_path, "r") as f:
        code = f.read()
        f.close()

    code = run_transcompile_code(code, source="BANG", target="CUDA")
    print(code)
