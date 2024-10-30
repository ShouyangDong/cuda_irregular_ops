import os
import subprocess

from benchmark.evaluation.cuda_test.result_test import run_test
from smt.loop_transformation.loop_recovery import ast_loop_recovery
from smt.tensorization.detensorization import ast_detensorization
from src.pre_processing.preprocessing import run_detensorization, run_loop_recovery


def unitest(file_name, code, target):
    base_name = os.path.basename(file_name)
    # save code as file
    if target == "CUDA":
        with open(base_name, mode="w") as f:
            f.write(code)
            f.close()
        success, output = run_test(base_name, ".benchmark/data//cuda_test/test_add.py")
        _ = subprocess.run(["rm", base_name])
        return success
    return False


def falcon_preprocess_pipeline(file_name, target):
    with open(file_name, "r") as f:
        org_code = f.read()
        f.close()

    device_code = org_code.split("extern")[0]
    host_code = "extern" + org_code.split("extern")[1]

    code = run_loop_recovery(device_code, target)
    if not unitest(file_name, device_code + host_code, target):

        code = ast_loop_recovery(device_code, target)

    if target in ["BANG"]:
        modi_code = run_detensorization(code, target)
        if not unitest(file_name, modi_code + host_code, target):
            modi_code = ast_detensorization(code, target)

    return code


if __name__ == "__main__":
    cuda_file_name = "benchmark/data/mlu_code_test/sign_45_25.mlu"
    code = falcon_preprocess_pipeline(cuda_file_name, target="BANG")
    print(code)

    cuda_file_name = "benchmark/data/cuda_code_test/add_3_3_256.cu"
    code = falcon_preprocess_pipeline(cuda_file_name, target="CUDA")
    print(code)
