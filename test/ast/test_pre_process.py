from smt.loop_transformation.loop_recovery import ast_loop_recovery
from smt.simplification import simplify_code
from smt.stmt_simplification import ast_stmt_simplification
from smt.tensorization.detensorization import ast_detensorization
from smt.loop_inline import ast_inline

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
