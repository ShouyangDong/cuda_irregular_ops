import re
import json


def inline_function(file_path, func_name, code):
    """Traverse the AST and inline small functions"""
    with open(file_path) as json_file:
        function_definitions = json.load(json_file)
    # Get the function definition from the json file.
    func_definition = function_definitions[func_name]

    # 替换函数调用和函数名
    intrinsic_bodys = []
    for string in code.split(func_name + "(")[1:]:
        intrinsic_body = func_name + "(" + string.split(");")[0]
        intrinsic_bodys.append(intrinsic_body)

    # get the arguments of intrinsic function
    intrinsic_args = []
    for intrinsic in intrinsic_bodys:
        intrinsic_arg = (
            intrinsic.split(func_name + "(")[1].split(");")[0].strip().split(",")
        )
        intrinsic_args.append(intrinsic_arg)

    definition_args = (
        func_definition.split(func_name + "(")[1].split(");")[0].strip().split(",")
    )
    for index, intrinsic_body in enumerate(intrinsic_bodys):
        parameter_mappings = {}
        for key, value in zip(definition_args, intrinsic_args[index]):
            parameter_mappings[key] = value

        function_body = func_definition.split("->")[1]
        # replace the arguments
        for param in parameter_mappings:
            function_body = function_body.replace(param, parameter_mappings[param])
        # replace the intrinsic with C sequential code
        code = code.replace(intrinsic_body, function_body)
    return code


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
        code = inline_function(file_path, func_name, code)
    print("[INFO]******************output: ", code)
