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
    extern "C" __global__ void __launch_bounds__(1024) add_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ T_add) {
    if (((blockIdx.x * 1024) + (threadIdx.x)) < 4096) {
        T_add[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (A[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] + B[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
    }
    }
    """
    code = run_transcompile_code(cuda_code, source="CUDA", target="BANG")
    print(code)
