import json
import os
import subprocess

from benchmark.evaluation.cuda_test.result_test import run_test as cuda_run_test
from benchmark.evaluation.mlu_test.result_test import run_test as bang_run_test
from smt.auto_cache import ast_auto_cache
from smt.thread_binding import ast_thread_binding
from src.post_processing.post_processing import (
    replace_operation_with_intrinsic,
    run_cache_process,
    run_code_decoration,
    run_tensorization,
    run_thread_binding,
)


def unitest(file_name, code, target):
    base_name = os.path.basename(file_name)
    with open(base_name, mode="w") as f:
        f.write(code)
        f.close()
    # save code as file
    if target == "CUDA":
        success, output = cuda_run_test(
            base_name, ".benchmark/evaluation/cuda_test/test_add.py"
        )
        _ = subprocess.run(["rm", base_name])
        return success
    elif target == "BANG":
        success, output = bang_run_test(
            base_name, ".benchmark/evaluation/mlu_test/test_add.py"
        )
        _ = subprocess.run(["rm", base_name])
        return success
    return False


def falcon_postprocess_pipeline(code, file_name, target):
    if target == "CUDA":
        host_code = """
        extern "C" void add_kernel(float *C, float *A, float *B, int size) {
        float *d_A, *d_B, *d_C;

        cudaMalloc(&d_A, size * sizeof(float));
        cudaMalloc(&d_B, size * sizeof(float));
        cudaMalloc(&d_C, size * sizeof(float));

        cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, size * sizeof(float), cudaMemcpyHostToDevice);

        dim3 blockSize(1024);
        dim3 numBlocks(256);
        add<<<numBlocks, blockSize>>>(d_A, d_B, d_C);

        cudaMemcpy(C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        }
        """
    if target == "BANG":
        host_code = """
        extern "C" void add_kernel(float *C, float *A, float *B, int size) {
        cnrtQueue_t queue;
        cnrtSetDevice(0);
        cnrtQueueCreate(&queue);
        float *d_A, *d_B, *d_C;

        cnrtMalloc((void **)(&d_A), size * sizeof(float));
        cnrtMalloc((void **)(&d_B), size * sizeof(float));
        cnrtMalloc((void **)(&d_C), size * sizeof(float));

        cnrtMemcpy(d_A, A, size * sizeof(float), cnrtMemcpyHostToDev);
        cnrtMemcpy(d_B, B, size * sizeof(float), cnrtMemcpyHostToDev);

        cnrtDim3_t dim = {1, 4, 4};
        cnrtFunctionType_t ktype = CNRT_FUNC_TYPE_UNION4;

        add<<<dim, ktype, queue>>>(d_A, d_B, d_C);

        cnrtMemcpy(C, d_C, size * sizeof(float), cnrtMemcpyDevToHost);

        cnrtFree(d_A);
        cnrtFree(d_B);
        cnrtFree(d_C);
        }
        """

    final_code = run_thread_binding(code, target)
    if not unitest(file_name, final_code + host_code, target):
        final_code = ast_thread_binding(code, target)
    print("[INFO] final_code: ", final_code)
    # when target is "BANG" or "DLBOOST", insert tensorization process.
    if target in ["BANG", "DLBOOST"]:
        code = run_code_decoration(final_code)
        print("[INFO] decorate code: ", code)
        op_pragma = {}
        if target == "BANG":
            op_pragma = json.load(
                open("./documents/operation_bang_C_instruction_map.json", "r")
            )
        code, space_maps = replace_operation_with_intrinsic(code, op_pragma)
        cache_code = run_cache_process(code, space_maps)

        if not unitest(file_name, cache_code + host_code, target):
            cache_code = ast_auto_cache(code, space_maps)
        print("[INFO] decorate code: ", cache_code)
        code = run_code_decoration(cache_code)

        final_code = run_tensorization(code, target)
        if not unitest(file_name, final_code + host_code, target):
            final_code = ast_auto_cache(code, space_maps)
    return final_code


if __name__ == "__main__":
    code = """
    void add_kernel(float* output, float* input1, float* input2) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < 128; k++) {
                    for (int l = 0; l < 128; l++) {
                        output[i * 4 * 128 * 128 + j * 128 * 128 + k * 128 + l] = input1[i * 4 * 128 * 128 + j * 128 * 128 + k * 128 + l] + input2[i * 4 * 128 * 128 + j * 128 * 128 + k * 128 + l];
                    }
                }
            }
        }
    }
    """
    cuda_file_name = "./add_4_4_128_128."
    code = falcon_postprocess_pipeline(code, cuda_file_name + "mlu", target="BANG")
    print(code)

    code = falcon_postprocess_pipeline(code, cuda_file_name + "cu", target="CUDA")
    print(code)
