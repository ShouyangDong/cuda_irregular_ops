from falcon.smt.loop_inline import ast_inline
from falcon.smt.loop_transformation.loop_recovery import ast_loop_recovery
from falcon.smt.simplification import simplify_code
from falcon.smt.stmt_simplification import ast_stmt_simplification
from falcon.smt.tensorization.detensorization import ast_detensorization


def pre_processing_pipeline(code, target):
    code = ast_loop_recovery(code, target)
    code = ast_detensorization(code, target)
    code = simplify_code(code)
    code = ast_stmt_simplification(code)
    code = ast_inline(code)
    return code


if __name__ == "__main__":
    func_content = """
    extern "C" __mlu_global__ void tanh(float* input0, float* active_tanh_210) {
        __nram__ float input0_local_nram[640];
        __memcpy(((float *)input0_local_nram + (0)), ((float *)input0 + (((((int)clusterId) * 2560) + (((int)coreId) * 640)))), 2560, GDRAM2NRAM);
        __bang_active_tanh(((float *)input0_local_nram + (0)), ((float *)input0_local_nram + (0)), 640);
        __memcpy(((float *)active_tanh_210 + (((((int)clusterId) * 2560) + (((int)coreId) * 640)))), ((float *)input0_local_nram + (0)), 2560, NRAM2GDRAM);
    }
    """
    code = pre_processing_pipeline(func_content, target="BANG")
    print(code)

    func_content = """
    extern "C" __global__ void __launch_bounds__(1024) exp_kernel(float* __restrict__ A, float* __restrict__ compute) {
        if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 1125) {
            compute[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = __expf(A[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
        }
    }
    """
    code = pre_processing_pipeline(func_content, target="CUDA")
    print(code)
    code = """
    extern "C" __mlu_global__ void add(float* lhs, float* rhs, float* add_1935) {
        __nram__ float lhs_local_nram[2048];
        __memcpy(((float *)lhs_local_nram + (0)), ((float *)lhs + ((((int)coreId) * 1024))), 4096, GDRAM2NRAM);
        __memcpy(((float *)lhs_local_nram + (1024)), ((float *)rhs + ((((int)coreId) * 1024))), 4096, GDRAM2NRAM);
        __bang_add(((float *)lhs_local_nram + (0)), ((float *)lhs_local_nram + (0)), ((float *)lhs_local_nram + (1024)), 1024);
        __memcpy(((float *)add_1935 + ((((int)coreId) * 1024))), ((float *)lhs_local_nram + (0)), 4096, NRAM2GDRAM);
    }
    """
    code = pre_processing_pipeline(code, target="BANG")
    print(code)
