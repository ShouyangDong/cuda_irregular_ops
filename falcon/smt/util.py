import re

from pycparser import c_ast


class NodeTransformer(c_ast.NodeVisitor):
    def generic_visit(self, node):
        for field, old_value in iter_fields(node):
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, c_ast.Node):
                        value = self.visit(value)
                        if value is None:
                            continue
                        elif not isinstance(value, c_ast.Node):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, c_ast.FuncCall):
                new_node = self.generic_visit(old_value)
                setattr(node, field, new_node)

            elif isinstance(old_value, c_ast.Node):
                new_node = self.visit(old_value)
                setattr(node, field, new_node)
        return node


def iter_fields(node):
    # this doesn't look pretty because `pycparser` decided to have structure
    # for AST node classes different from stdlib ones
    index = 0
    children = node.children()
    while index < len(children):
        name, child = children[index]
        try:
            bracket_index = name.index("[")
        except ValueError:
            yield name, child
            index += 1
        else:
            name = name[:bracket_index]
            child = getattr(node, name)
            index += len(child)
            yield name, child


def add_memory_prefix(code):
    # Define the memory types and their associated prefixes
    prefix_map = {
        "_Nram": "__nram__ float",
        "_Wram": "__wram__ float",
        "_nram": "__nram__ float",
        "_wram": "__wram__ float",
    }

    # Regex pattern to match the variable declarations
    pattern = r"\bfloat\s+(\w+_(Nram|Wram|nram|wram|Gdram))\b"

    # Function to replace matched float declarations with the appropriate prefix
    def replacer(match):
        var_name = match.group(1)
        suffix = match.group(2)
        if f"_{suffix}" in prefix_map:
            return f"{prefix_map[f'_{suffix}']} {var_name}"
        return match.group(0)

    # Substitute in the code using regex
    modified_code = re.sub(pattern, replacer, code)
    return (
        "__mlu_global__ " + modified_code
        if "__mlu_global__ " not in modified_code
        else modified_code
    )


def add_parallel_variable_prefix(code):
    code = re.sub(r"threadIdxx", "threadIdx.x", code)
    code = re.sub(r"threadIdxy", "threadIdx.y", code)
    code = re.sub(r"threadIdxz", "threadIdx.z", code)
    code = re.sub(r"blockIdxx", "blockIdx.x", code)
    code = re.sub(r"blockIdxy", "blockIdx.y", code)
    code = re.sub(r"blockIdxz", "blockIdx.z", code)
    return "__global__ " + code if "__global__ " not in code else code


def remove_target_prefix(code, target):
    if target == "BANG":
        # 移除 `extern "C"`
        code = re.sub(r'extern "C"\s+', "", code)

        # 移除 `__global__` 修饰符
        code = re.sub(r"__mlu_global__\s+", "", code)

        # 使用正则表达式移除 `__nram__` 关键字，仅保留 `float` 声明
        code = re.sub(r"\b__nram__\s+", "", code)

        # 移除所有 C/C++ 样式的注释
        code = re.sub(r"//.*?\n|/\*.*?\*/", "", code, flags=re.S)

    elif target == "CUDA":
        # 移除 `extern "C"`
        code = re.sub(r'extern "C"\s+', "", code)

        # 移除 `__global__` 修饰符
        code = re.sub(r"__global__\s+", "", code)

        # 移除 `__launch_bounds__(\d+)`
        code = re.sub(r"__launch_bounds__\(\d+\)\s+", "", code)

        # 移除 `__launch_bounds__(\d+)`
        code = re.sub(r"\b__restrict__\b", "", code)

        # 移除所有 C/C++ 样式的注释
        code = re.sub(r"//.*?\n|/\*.*?\*/", "", code, flags=re.S)
    return code
