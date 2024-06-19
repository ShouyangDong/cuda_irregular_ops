from pycparser import c_parser, c_ast, c_generator


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

class CacheWriteTransformer(NodeTransformer):
    """
    A transformer that modifies loops to perform cache writes based on pragma directives.

    Attributes:
    - pragma_info (dict): A dictionary mapping nodes to pragma strings.
    - agrs (dict): A dictionary mapping buffer names to pragma strings.
    """
    def __init__(self, pragma_info, args):
        self.pragma_info = pragma_info
        self.agrs = args

    def visit_Pragma(self, node):
        """
        Handle pragma nodes by removing them since their effect is applied during the transformation.
        
        Parameters:
        - node: The AST node for the pragma.
        
        Returns:
        - None, to indicate the node should be removed from the AST.
        """
        return None

    def visit_For(self, node):
        """
        Transform the for loop to perform cache writes as specified by the pragma.
        
        Parameters:
        - node: The AST node for the for loop.
        
        Returns:
        - A list of new AST nodes that replace the original for loop.
        """
        # Check if the current loop corresponds to the axis we want to transform.
        if self.axis in (decl.name for decl in node.init.decls):
            # Retrieve the pragma string associated with this loop.
            pragma = self.pragma_info.get(node, "")
            memory_space = self.get_memory_space(pragma)

            # Create a new variable name for the cache.
            cache_var = f"{self.buffer_var}_nram"
            new_body = []

            # Replace the original buffer variable with the cache variable in the loop body.
            for stmt in node.body:
                if isinstance(stmt, SomeASTNodeType):  # Replace with actual node type check
                    stmt.value = stmt.value.replace(self.buffer_var, cache_var)
                new_body.append(stmt)

            # Create a new for loop node with the updated body.
            new_node = type(node)(node.init, node.cond, node.next, new_body)

            # Create a loop to copy data back from the cache to the original buffer.
            copy_back_node = self.create_copy_back_loop(self.buffer_var, cache_var, node)

            # Return both the cache write loop and the copy back loop.
            return [new_node, copy_back_node]
        else:
            return node

    def get_memory_space(self, pragma):
        """
        Extract the memory space from the pragma string.
        
        Parameters:
        - pragma (str): The pragma string containing memory space information.
        
        Returns:
        - The extracted memory space as a string.
        """
        return pragma.split("output[")[1].split("]")[0]


class PragmaVisitor(c_ast.NodeVisitor):
    def __init__(self):
        self.pragma_info = {}
        self.args = {}

    def visit_Compound(self, node):
        # Get the block_items
        blocks = node.block_items
        for index, node in enumerate(blocks):
            if isinstance(node, c_ast.Pragma):
                self.pragma_info[blocks[index + 1]] = node.string
                stage_visitor = StageVisitor()
                stage_visitor.visit(blocks[index + 1])
                self.args[node.string] = stage_visitor.write_args

class StageVisitor(c_ast.NodeVisitor):
    def __init__(self):
        self.write_args = []

    def visit_Assignment(self, node):
        self.write_args.append(node.lvalue)

if __name__ == "__main__":
    code = """
        void __bang_add(C, A, B, size) {
            #pragma __bang_add(input[Nram, Nram], output[Nram])
            for (int i_add = 0; i_add < size; i_add++) {
                C[i_add] = A[i_add] + B[i_add];
            }
        }
    """
    parser = c_parser.CParser()
    ast = parser.parse(code)
    visitor = PragmaVisitor()
    visitor.visit(ast)
    pragma_info = visitor.pragma_info
    args = visitor.args
    print("[INFO]****************info: ", pragma_info)
    print("[INFO]****************args: ", args)