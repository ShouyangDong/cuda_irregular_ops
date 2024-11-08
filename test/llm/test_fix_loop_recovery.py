from falcon.src.pre_processing.preprocessing import run_loop_recovery
from falcon.unit_test import unit_test


def transform_loop_recovery(code, target):
    code = run_loop_recovery(code, target)
    if unit_test(code):
        code = smt_fix(code, target)
    return code


if __name__ == "__main__":
    code = """
    extern "C" __global__ void __launch_bounds__(1024) add_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ T_add) {
        if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 2309) {
            T_add[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (A[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] + B[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
        }
    }
    """
    code = run_loop_recovery(code, target="CUDA")
    print(code)

    code = """
    extern "C" __mlu_global__ void multiply(float* A_nram, float* B_wram, float* C_nram) {
        for (int col = 0; col < 64; col++) {
            C_nram[(clusterId * 4 + coreId) * 64 + col] = 0.0f;
            for (int i = 0; i < 512; i++) {
                C_nram[col] += A_nram[i] * B_wram[i * 64 + col];
            }
        }
    }
    """
    code = run_loop_recovery(code, target="BANG")
    print(code)
