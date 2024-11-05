from src.loop_transformation.loop_transformation import (
    run_apply_split,
    run_loop_fusion,
    run_split_annotation,
)
from src.post_processing.post_processing import post_processing_pipeline
from src.pre_processing.preprocessing import pre_processing_pipeline


def run_transcompile_code(code, source, target):
    code = pre_processing_pipeline(code, target=source)
    code = run_loop_fusion(code)
    code = run_split_annotation(code)
    code = run_apply_split(code)
    code = post_processing_pipeline(code, target)
    return code


if __name__ == "__main__":
    cuda_code = """
    extern "C" __mlu_global__ void add(float* lhs, float* rhs, float* add_1935) {
    __nram__ float lhs_local_nram[2048];
    __memcpy(((float *)lhs_local_nram + (0)), ((float *)lhs + ((((int)coreId) * 1024))), 4096, GDRAM2NRAM);
    __memcpy(((float *)lhs_local_nram + (1024)), ((float *)rhs + ((((int)coreId) * 1024))), 4096, GDRAM2NRAM);
    __bang_add(((float *)lhs_local_nram + (0)), ((float *)lhs_local_nram + (0)), ((float *)lhs_local_nram + (1024)), 1024);
    __memcpy(((float *)add_1935 + ((((int)coreId) * 1024))), ((float *)lhs_local_nram + (0)), 4096, NRAM2GDRAM);
    }
    """
    code = run_transcompile_code(cuda_code, source="BANG", target="CUDA")
    print(code)
