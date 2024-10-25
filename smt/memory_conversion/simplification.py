import re


def merge_conditions(code: str) -> str:
    # 定义正则表达式，匹配重复的条件判断和代码块
    pattern = (
        r"if \(\(\(\(int\)clusterId\) \* 4\) \+ \(\(int\)coreId\)\) < 15\) \{([^}]+)\}"
    )

    # 找到所有匹配的代码块
    matches = re.findall(pattern, code)

    # 如果找到的代码块一致，则可以进行合并
    if len(matches) > 1 and all(
        block.strip() == matches[0].strip() for block in matches
    ):
        merged_block = matches[0].strip()

        # 构建合并后的代码
        merged_code = re.sub(pattern, "", code)  # 去掉原来的所有重复块
        merged_condition = f"if ((((int)clusterId) * 4) + ((int)coreId)) < 15) {{\n{merged_block}\n}}\n"

        # 将合并后的代码块插入到代码的开始部分
        merged_code = merged_condition + merged_code
        return merged_code

    return code


# 原始代码
code = """
void add_kernel0(float* lhs, float* rhs, float* add_1515) {
    float lhs_local_nram[128];
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

# 进行条件合并
merged_code = merge_conditions(code)
print(merged_code)
