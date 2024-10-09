from pycparser import c_ast, c_parser
import re

ParaVar = {"threadIdx.x": 1024, "blockIdx.x": 256, "coreId": 4, "clusterId": 4}

cuda_paravar = ["threadIdx.x", "threadIdx.y", "blockIdx.x", "blockIdx.y"]
mlu_paravar = ["coreId", "clusterId"]


def get_thread_dim(cuda_code):
    """The re module in Python is used to write a regular expression
    that matches the number inside the parentheses."""
    match = re.search(r"__launch_bounds__\((\d+)\)", cuda_code)
    if match:
        # 打印匹配的数值
        launch_bounds_value = int(match.group(1))
        return launch_bounds_value
    else:
        return None


class LoopRecoveryVisitot(c_ast.NodeVisitor):
    def visit_If(self, node):
        return None


if __name__ == "__main__":
    cuda_code = """
    __global__ void __launch_bounds__(1024) add(float* __restrict__ A, float* __restrict__ B, float* __restrict__ T_add) {
        if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 2309) {
            T_add[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (A[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] + B[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
        }
    }
    """
    dim = get_thread_dim(cuda_code)
    print(f"The value inside __launch_bounds__ is: {dim}")

    cuda_code = """
    void add(float*  A, float*  B, float*  T_add) {
        if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < 2309) {
            T_add[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (A[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] + B[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);
        }
    }
    """
    parser = c_parser.CParser()
    ast = parser.parse(cuda_code)
    # print(ast)
