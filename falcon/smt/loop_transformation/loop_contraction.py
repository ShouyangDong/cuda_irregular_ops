from pycparser import c_ast, c_generator, c_parser

from falcon.simplification import simplify_code
from falcon.util import NodeTransformer, remove_target_prefix


class LoopNestFusionVisitor(NodeTransformer):
    """
    把两段相同循环域的 for-loop 融合成一个：
    for(...) {   // outer1
      for(...) { /* body1 */ }
    }
    for(...) {   // outer2，与 outer1 的 init/cond/next 相同
      for(...) { /* body2 */ }
    }
    变成：
    for(...) {
      for(...) { /* body1 */ }
      for(...) { /* body2 */ }
    }
    """

    def visit_Compound(self, node):
        if not node.block_items:
            return node
        new_items = []
        i = 0
        while i < len(node.block_items):
            stmt = node.block_items[i]
            # 检测当前是 For 且下一个也是 For
            if (
                isinstance(stmt, c_ast.For)
                and i + 1 < len(node.block_items)
                and isinstance(node.block_items[i + 1], c_ast.For)
            ):
                outer1 = stmt
                outer2 = node.block_items[i + 1]
                # 比对两个 outer-loop 的 init/cond/next 是否一致
                if self._same_loop_header(outer1, outer2):
                    # 融合：把 outer2.body 放到 outer1.body 后面
                    fused = self._fuse_loops(outer1, outer2)
                    new_items.append(self.visit(fused))
                    i += 2
                    continue
            # 否则正常保留
            new_items.append(self.visit(stmt))
            i += 1

        node.block_items = new_items
        return node

    def _same_loop_header(self, f1: c_ast.For, f2: c_ast.For) -> bool:
        # 简单地把 init/cond/next 的文本化比较
        return (
            self._node_to_str(f1.init) == self._node_to_str(f2.init)
            and self._node_to_str(f1.cond) == self._node_to_str(f2.cond)
            and self._node_to_str(f1.next) == self._node_to_str(f2.next)
        )

    def _fuse_loops(self, outer1: c_ast.For, outer2: c_ast.For) -> c_ast.For:
        # 把 outer2.body.block_items 并入 outer1.body
        body1 = outer1.stmt
        body2 = outer2.stmt
        # 确保两个 body 都是 Compound，否则包一层
        if not isinstance(body1, c_ast.Compound):
            body1 = c_ast.Compound([body1])
        if not isinstance(body2, c_ast.Compound):
            body2 = c_ast.Compound([body2])
        # 生成新的 body
        fused_body = c_ast.Compound(body1.block_items + body2.block_items)
        outer1.stmt = fused_body
        return outer1

    def _node_to_str(self, node):
        """把 AST 节点转字符串，方便比较"""
        if node is None:
            return ""
        return c_generator.CGenerator().visit(node)


def ast_loop_contraction(c_code):
    """Start to run loop contraction."""
    # 1. 解析
    c_code = remove_target_prefix(c_code)
    parser = c_parser.CParser()
    ast = parser.parse(c_code)

    # 2. 转换（融合循环）
    visitor = LoopNestFusionVisitor()
    visitor.visit(ast)

    # 3. 生成 C 代码
    generator = c_generator.CGenerator()
    code = generator.visit(ast)
    code = simplify_code(code)
    return code


if __name__ == "__main__":
    code = r"""
  void kernel(float A[N][M], float B[N][M], float C[N][M], float D[N][M]) {
    for(int i = 0; i < N; i++) {
      for(int j = 0; j < M; j++) {
        A[i][j] = B[i][j] + C[i][j];
      }
    }

    for(int i = 0; i < N; i++) {
      for(int j = 0; j < M; j++) {
        D[i][j] = A[i][j] * 2;
      }
    }
  }
  """
    code = ast_loop_contraction(code)
    print(code)
