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
    func_name = "__memcpy"
    output_code = inline_function(file_path, func_name, code)