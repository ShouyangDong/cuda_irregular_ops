from smt.loop_transformation.loop_recovery import ast_loop_recovery
from smt.tensorization.detensorization import ast_detensorization
from src.loop_transformation.loop_transformation import (
    run_apply_split,
    run_loop_fusion,
    run_split_annotation,
)
from src.post_processing.post_processing import post_processing_pipeline
from src.pre_processing.preprocessing import run_detensorization, run_loop_recovery


def run_transcompile_code(code, source, target):
    device_code = code.split("extern")[0]
    host_code = "extern" + code.split("extern")[1]

    code = run_loop_recovery(device_code, target)
    if not unitest(file_name, device_code + host_code, target):
        code = ast_loop_recovery(device_code, target)

    if target in ["BANG"]:
        modi_code = run_detensorization(code, target)
        if not unitest(file_name, modi_code + host_code, target):
            modi_code = ast_detensorization(code, target)

    code = run_loop_fusion(code)
    code = run_split_annotation(code)
    code = run_apply_split(code)
    code = post_processing_pipeline(code, target)
    return code


if __name__ == "__main__":
    file_path = "benchmark/data/mlu_code_test/add_4_4_4_64.mlu"
    with open(file_path, "r") as f:
        code = f.read()
        f.close()

    code = run_transcompile_code(code, source="BANG", target="CUDA")
    print(code)
