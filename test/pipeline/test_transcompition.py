from falcon.smt.loop_transformation.loop_fusion import ast_loop_fusion
from falcon.smt.loop_transformation.loop_recovery import ast_loop_recovery
from falcon.smt.loop_transformation.loop_split import ast_loop_split
from falcon.smt.tensorization.detensorization import ast_detensorization
from falcon.smt.thread_binding import ast_thread_binding
from falcon.src.loop_transformation.loop_transformation import (
    run_apply_split,
    run_loop_fusion,
    run_split_annotation,
)
from falcon.src.post_processing.post_processing import run_thread_binding
from falcon.src.pre_processing.preprocessing import (
    run_detensorization,
    run_loop_recovery,
)
from falcon.unit_test import unit_test


def run_transcompile_code(file_name, source, target):
    with open(file_name, "r") as f:
        device_code = f.read()
        f.close()

    # preprocess
    code = run_loop_recovery(device_code, target)
    if not unit_test(file_name, device_code):
        code = ast_loop_recovery(device_code, source)

    print("[INFO]*********loop recovery: ", code)
    if source in ["BANG"]:
        try:
            modi_code = run_detensorization(code, source)
        except:
            modi_code = None

        if not unit_test(file_name, modi_code):
            modi_code = ast_detensorization(code, source)

    print("[INFO]***********detensorization: ", modi_code)
    # loop transformation
    fusion_code = run_loop_fusion(modi_code)
    if not unit_test(file_name, fusion_code):
        fusion_code = ast_loop_fusion(modi_code)

    print("[INFO]***********fusion: ", fusion_code)
    code = run_split_annotation(fusion_code)
    print("[INFO]***********split annotate: ", code)
    split_code = run_apply_split(code)
    print("[INFO]***********split: ", code)
    if not unit_test(file_name, split_code):
        split_code = ast_loop_split(code)
    print("[INFO]***********split: ", split_code)
    # postprocessing
    final_code = run_thread_binding(split_code, target)
    if not unit_test(file_name, final_code):
        final_code = ast_thread_binding(split_code, target)

    return final_code


if __name__ == "__main__":
    file_name = "benchmark/data/mlu_code_test/add_4_4_4_64.mlu"
    code = run_transcompile_code(file_name, source="BANG", target="CUDA")
    print(code)
