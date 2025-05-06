import os
import re

from pycparser import c_ast, c_generator, parse_file


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
    pattern = re.compile(
        r"(?<!__nram__\s)(?<!__wram__\s)float\s+"
        r"(\w+_(?:Nram|Wram|nram|wram|Gdram))\b"
    )

    # Function to replace matched float declarations with the appropriate
    # prefix
    def replacer(match):
        var_name = match.group(1)  # 变量名，如 "lhs_local_Nram"
        suffix = "_" + match.group(1).split("_")[-1]  # "_Nram"
        # 如果在映射里，就替换
        if suffix in prefix_map:
            return f"{prefix_map[suffix]} {var_name}"
        # 否则保留原样
        return match.group(0)

    # Substitute in the code using regex
    modified_code = pattern.sub(replacer, code)
    if "memcpy" in modified_code and "__memcpy" not in modified_code:
        modified_code = modified_code.replace("memcpy", "__memcpy")

    if "extern" in code:
        return re.sub(r'(extern\s+"C"\s+)(void)', r"\1__mlu_global__ \2", code)
    else:
        return 'extern "C" __mlu_global__ ' + code


def add_parallel_variable_prefix(code):
    code = re.sub(r"threadIdxx", "threadIdx.x", code)
    code = re.sub(r"threadIdxy", "threadIdx.y", code)
    code = re.sub(r"threadIdxz", "threadIdx.z", code)
    code = re.sub(r"blockIdxx", "blockIdx.x", code)
    code = re.sub(r"blockIdxy", "blockIdx.y", code)
    code = re.sub(r"blockIdxz", "blockIdx.z", code)
    return "__global__ " + code if "__global__ " not in code else code


def remove_target_prefix(code, target=None):
    patterns = [
        (r'extern "C"\s+', ""),  # 移除 `extern "C"`
        (r"__mlu_global__\s+", ""),  # 移除 `__mlu_global__`
        (r"\b__nram__\s+", ""),  # 移除 `__nram__`
        (r"\b__wram__\s+", ""),  # 移除 `__wram__`
        (r"__global__\s+", ""),  # 移除 `__global__`
        (r"__launch_bounds__\(\d+\)\s+", ""),  # 移除 `__launch_bounds__`
        (r"\b__restrict__\b", ""),  # 移除 `__restrict__`
        (r"//.*?\n|/\*.*?\*/", "", re.S),  # 移除所有 C/C++ 注释
        (r"\bthreadIdxx\b", "threadIdxx"),  # 改为下划线风格
        (r"\bthreadIdxy\b", "threadIdxy"),
        (r"\bthreadIdxz\b", "threadIdxz"),
        (r"\bblockIdxx\b", "blockIdxx"),
        (r"\bblockIdxy\b", "blockIdxy"),
        (r"\bblockIdxz\b", "blockIdxz"),
        (
            r"static_cast<\s*([A-Za-z_][A-Za-z0-9_]*)\s*>\s*\(([^)]+?)\)",
            r"(\1)(\2)",
        ),
        (r"reinterpret_cast<\s*([^>]+?)\s*>\s*\(([^)]+?)\)", r"(\1)(\2)"),
        # 处理 wmma 命名空间的类型（模板语法 -> C struct 类型）
        (r"wmma::fragment<[^>]+?>", "wmma_fragment"),
        # 处理 wmma::前缀函数调用（如 wmma::load_matrix_sync）
        (r"\bwmma::(\w+)", r"wmma_\1"),
    ]

    # 遍历模式列表，应用替换
    for pattern, replacement, *flags in patterns:
        code = re.sub(
            pattern, replacement, code, flags=flags[0] if flags else 0
        )

    headers = [
        {
            "header": '#include "stdint.h"',
            "trigger_keywords": ["int8_t", "int32_t"],
        },
        {
            "header": '#include "simd.h"',
            "trigger_keywords": ["__m128i", "_mm_"],
        },
        {
            "header": '#include "simd_cuda.h"',
            "trigger_keywords": [
                "wmma_fragment",
                "wmma_fill_fragment",
                "wmma_load_matrix_sync",
                "wmma_mma_sync",
                "wmma_store_matrix_sync",
            ],
        },
    ]
    lines = code.splitlines()
    existing_includes = set(
        line.strip() for line in lines if line.strip().startswith("#include")
    )

    added_headers = []
    for h in headers:
        needs = any(kw in code for kw in h["trigger_keywords"])
        has_header = h["header"] in existing_includes
        if needs and not has_header:
            added_headers.append(h["header"])

    if added_headers:
        return "\n".join(added_headers) + "\n\n" + code
    else:
        if "half" in code:
            code = '#include "stdhalf.h"' + "\n\n" + code
        return code


def get_target(code, target=None):
    # 判断文件类型并设置目标
    if (
        "__mlu_global" in code
        or "__bang" in code
        or "coreId" in code
        or "__nram__" in code
    ):
        target, file_type = "mlu", ".mlu"
    elif target == "hip" and ("__global__" in code or "threadIdx.x" in code):
        target, file_type = "hip", ".hip"
    elif "__global__" in code or "threadIdx.x" in code or "wmma" in code:
        target, file_type = "cuda", ".cu"
    else:
        target, file_type = "cpu", ".cpp"
    return target, file_type


def make_full_func(code, target=None):
    target, file_type = get_target(code, target)
    if target == "mlu":
        code = add_memory_prefix(code)
    elif target in ["cuda", "hip"]:
        code = add_parallel_variable_prefix(code)
    return code


def parse_code_ast(code, target=None):
    code = remove_target_prefix(code, target=target)
    filename = "./local_parse_test.c"
    with open(filename, "w") as f:
        f.write(code)
    ast = parse_file(
        filename,
        use_cpp=True,
        cpp_path="cpp",
        cpp_args=["-Iutils/fake_libc_include"],
    )
    os.remove(filename)
    return ast


def generate_code(ast):
    generator = c_generator.CGenerator()
    # Collect all function definitions in the translation unit
    func_defs = [ext for ext in ast.ext if isinstance(ext, c_ast.FuncDef)]
    # Generate code for each function and join them with two newlines
    all_functions_code = "\n\n".join(
        generator.visit(func) for func in func_defs
    )
    return all_functions_code
