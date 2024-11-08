import os
import subprocess

from benchmark.evaluation.cuda_test.result_test import run_test as cuda_run_test
from benchmark.evaluation.mlu_test.result_test import run_test as bang_run_test
from falcon.smt.loop_transformation.loop_recovery import ast_loop_recovery
from falcon.smt.tensorization.detensorization import ast_detensorization
from falcon.src.pre_processing.preprocessing import run_detensorization, run_loop_recovery
from falcon.unit_test import unit_test

def falcon_preprocess_pipeline(file_name, target):
    with open(file_name, "r") as f:
        org_code = f.read()
        f.close()

    device_code = org_code.split("extern")[0]
    host_code = "extern" + org_code.split("extern")[1]
    print(device_code)
    # code = run_loop_recovery(device_code, target)
    # print("[INFO]******code: ", code)
    # if not unit_test(file_name, device_code + host_code):
    #     code = ast_loop_recovery(device_code, target)

    # if target in ["BANG"]:
    #     modi_code = run_detensorization(code, target)
    #     if not unit_test(file_name, modi_code + host_code):
    #         modi_code = ast_detensorization(code, target)

    return device_code


if __name__ == "__main__":
    bang_file_name = "benchmark/data/mlu_code_test/sign_45_25.mlu"
    with open(bang_file_name, "r") as f:
        code = f.read()
        f.close()
    result = unit_test(bang_file_name, code)
    # code = falcon_preprocess_pipeline(bang_file_name, target="BANG")
    print(result)

    # cuda_file_name = "benchmark/data/cuda_code_test/add_3_3_256.cu"
    # code = falcon_preprocess_pipeline(cuda_file_name, target="CUDA")
    # print(code)
