from src.pre_processing.preprocessing import pre_processing_pipeline


def run_transcompile_code(code, source, target):
    code = pre_processing_pipeline(code, target=source)
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
