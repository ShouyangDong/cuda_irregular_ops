from src.pre_processing.preprocessing import pre_processing_pipeline


def run_transcompile_code(code, source, target):
    code = pre_processing_pipeline(code, target=source)
    return code


if __name__ == "__main__":
    cuda_code = """
    extern "C" __mlu_global__ void add_kernel0(float* lhs, float* rhs, float* add_1935) {
    __nram__ float lhs_local_nram[2048];
    __memcpy(((float *)lhs_local_nram + (0)), ((float *)lhs + ((((int)coreId) * 1024))), 4096, GDRAM2NRAM);
    __memcpy(((float *)lhs_local_nram + (1024)), ((float *)rhs + ((((int)coreId) * 1024))), 4096, GDRAM2NRAM);
    __bang_add(((float *)lhs_local_nram + (0)), ((float *)lhs_local_nram + (0)), ((float *)lhs_local_nram + (1024)), 1024);
    __memcpy(((float *)add_1935 + ((((int)coreId) * 1024))), ((float *)lhs_local_nram + (0)), 4096, NRAM2GDRAM);
    }
    """
    code = run_transcompile_code(cuda_code, source="BANG", target="CUDA")
    print(code)
