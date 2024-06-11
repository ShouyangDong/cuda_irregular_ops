import re

def inline_function(file_path, func_name, code):
    """Traverse the AST and inline small functions"""
    with open(file_path, 'r') as f:
        function_definitions = f.read()


    # Get the function definition from the json file.
    func_definition = function_definitions[func_name]

    # replace the intrinsic with C sequential code
    # 替换函数调用和函数名
    code = code.replace(f"{func_name}(", func_definition)
    code = code.replace(f"def {func_name}", f"def __{func_name}")


    # get the arguments of intrinsic function
    intrinsic_args = code.split(func_name + "(")[1].splti(");")[0].strip().split(",")

    definition_args = func_definition.split(func_name + "(")[1].splti(");")[0].strip().split(",")

    for key, value in zip(intrinsic_args, definition_args):
        parameter_mappings[key] = value

    # 进行文本替换，将函数调用替换为C++函数的代码
    sequential_code = code.replace(f"{func_name}(", func_definition)
    # 替换参数名
    for param in parameter_mappings:
        sequential_code = re.sub(rf"\b{param}\b", parameter_mappings[param], code)
    return sequential_code


if __name__ == "__main__":
    file_path = "./function_intrinsic.json"
    code = """
        extern "C" __mlu_global__ void add_kernel0(float* lhs, float* rhs, float* add_1515) {
    __nram__ float lhs_local_nram[128];
    if (((((int)clusterId) * 4) + ((int)coreId)) < 15) {
        __memcpy(((float *)lhs_local_nram + (0)), ((float *)lhs + (((((int)clusterId) * 256) + (((int)coreId) * 64)))), 256, GDRAM2NRAM);
    }
    if (((((int)clusterId) * 4) + ((int)coreId)) < 15) {
        __memcpy(((float *)lhs_local_nram + (64)), ((float *)rhs + (((((int)clusterId) * 256) + (((int)coreId) * 64)))), 256, GDRAM2NRAM);
    }
    if (((((int)clusterId) * 4) + ((int)coreId)) < 15) {
        __bang_add(((float *)lhs_local_nram + (0)), ((float *)lhs_local_nram + (0)), ((float *)lhs_local_nram + (64)), 64);
    }
    if (((((int)clusterId) * 4) + ((int)coreId)) < 15) {
        __memcpy(((float *)add_1515 + (((((int)clusterId) * 256) + (((int)coreId) * 64)))), ((float *)lhs_local_nram + (0)), 256, NRAM2GDRAM);
    }
    }
    """

    func_names = ["__memcpy", "__bang_add"]
    file_path = "./function_definition.json"
    for func_name in func_names:
        output_code = inline_function(file_path, func_name, code)
    print("[INFO]***************output code: ", output_code)