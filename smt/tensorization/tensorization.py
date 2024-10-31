import re

import numpy as np
from pycparser import c_ast, c_generator, c_parser

from smt.util import NodeTransformer


class PragmaToSIMDTransformer(NodeTransformer):
    def __init__(self):
        self.loop_exts = []

    def visit_Compound(self, node):
        """遍历代码块，查找并修改带有 #pragma 的 for 循环"""
        new_block_items = []
        i = 0

        while i < len(node.block_items):
            stmt = node.block_items[i]

            # 检查是否是 #pragma operation
            if isinstance(stmt, c_ast.Pragma):
                pragma_text = stmt.string
                # 找到 pragma 下的下一个 for 循环
                if i + 1 < len(node.block_items) and isinstance(
                    node.block_items[i + 1], c_ast.For
                ):
                    transformed_stmt = self.transform_pragma_loop(
                        node.block_items[i + 1], pragma_text
                    )
                    new_block_items.append(transformed_stmt)  # 替换 for 循环为目标指令
                    i += 2  # 跳过下一个 for 循环，因为已经处理
                    continue
                else:
                    # 保留 #pragma 注解
                    new_block_items.append(stmt)
            else:
                # 保留未处理的语句
                new_block_items.append(stmt)

            i += 1

        node.block_items = new_block_items
        return node

    def transform_pragma_loop(self, for_loop, pragma_text):
        def get_args(pragma_text):
            # 使用正则表达式匹配 input 和 output 参数
            pattern = r"input\[(.*?)\].*?output\[(.*?)\]"
            match = re.search(pattern, pragma_text)

            if match:
                # 提取 input 和 output 参数
                input_params = match.group(1).split(", ")
                output_params = match.group(2).split(", ")
                return output_params, input_params
            else:
                return None, None

        output_params, input_params = get_args(pragma_text)
        assert output_params is not None
        assert input_params is not None
        args = output_params + input_params
        # 重置上界列表并访问 for_loop 以填充 self.loop_exts
        self.loop_exts = []
        self.visit(for_loop)  # 递归访问 for_loop 的子节点以触发 visit_For
        """根据 pragma 生成对应的指令，替换 for 循环"""
        if "memory" in pragma_text:
            src_dir = (
                input_params[0].split("_")[1].upper()
                if "_" in input_params[0]
                else "GDRAM"
            )
            dst_dir = (
                output_params[0].split("_")[1].upper()
                if "_" in output_params[0]
                else "GDRAM"
            )
            direction = src_dir + "2" + dst_dir
            transformed_code = c_ast.FuncCall(
                name=c_ast.ID("__memcpy"),
                args=c_ast.ExprList(
                    [
                        *([c_ast.ID(arg) for arg in args]),
                        c_ast.Constant(
                            "int",
                            str(4 * np.prod([int(ext) for ext in self.loop_exts])),
                        ),
                        c_ast.ID(direction),
                    ]
                ),
            )
        elif "matmul" in pragma_text:
            transformed_code = c_ast.FuncCall(
                name=c_ast.ID("__matmul"),
                args=c_ast.ExprList(
                    [c_ast.ID("C_nram"), c_ast.ID("A_nram"), c_ast.ID("B_wram")]
                ),
            )
        else:
            # 如果 pragma 不匹配，返回原始 for 循环
            return for_loop

        # 将函数调用包装在一个代码块中
        return transformed_code

    def visit_For(self, node):
        """提取 for 循环的上界值并存储在 self.loop_exts 列表中"""
        if isinstance(node.cond, c_ast.BinaryOp) and node.cond.op == "<":
            if isinstance(node.cond.right, c_ast.Constant):
                upper_bound = node.cond.right.value
                self.loop_exts.append(upper_bound)
        # 递归访问嵌套的 for 循环
        if isinstance(node.stmt, c_ast.Compound):
            for stmt in node.stmt.block_items:
                if isinstance(stmt, c_ast.For):
                    self.visit(stmt)  # 手动调用 visit 以访问嵌套的 for 循环
        return node


def ast_tensorization(code, target="BANG"):
    # 解析代码
    parser = c_parser.CParser()
    ast = parser.parse(code)

    # 进行 PragmaToSIMD 转换
    transformer = PragmaToSIMDTransformer()
    ast = transformer.visit(ast)

    # 输出修改后的代码
    generator = c_generator.CGenerator()
    return generator.visit(ast)


if __name__ == "__main__":
    # 示例代码
    code = """
    void matmul(float *A, float *B, float *C)
    {
    float B_wram[32768];
    float A_nram[512];
    float C_nram[64];
    
    #pragma operation(memory(input[B], output[B_wram]))
    for (int col = 0; col < 64; col++) {
        for (int i = 0; i < 512; i++) {
            B_wram[i * 64 + col] = B[i * 64 + col];
        }
    }

    #pragma operation(memory(input[A], output[A_wram]))
    for (int i = 0; i < 512; i++) {
        A_nram[i] = A[(clusterId * 4 + coreId) * 512 + i];
    }

    #pragma operation(matmul(input[A_nram, B_wram], output[C_nram]))
    for (int col = 0; col < 64; col++) {
        C_nram[(clusterId * 4 + coreId) * 64 + col] = 0.0f;
        for (int i = 0; i < 512; i++) {
            C_nram[col] += A_nram[i] * B_wram[i * 64 + col];
        }
    }

    #pragma operation(memory(input[C_nram], output[C]))
    for (int col = 0; col < 64; col++) {
        C[(clusterId * 4 + coreId) * 64 + col] = C_nram[col];
    }
    }
    """

    output_code = ast_tensorization(code)
    print(output_code)
