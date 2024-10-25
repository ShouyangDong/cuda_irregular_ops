from pycparser import c_ast, c_generator, c_parser

from smt.util import NodeTransformer


class CacheWriteTransformer(NodeTransformer):
    """
    A transformer that modifies loops to perform cache writes based on pragma directives.

    Attributes:
    - pragma_info (dict): A dictionary mapping nodes to pragma strings.
    - args (dict): A dictionary mapping buffer names to pragma strings.
    """

    def __init__(self, pragma_info, args):
        self.pragma_info = pragma_info
        self.args = args

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
        # TODO: maybe use list instead of dictionary
        for pragma, buffer_store_node in self.args.items():
            # TODO: only deal with single element list.
            buffer_var = buffer_store_node[0].name.name
            loop_index = buffer_store_node[0].subscript.name
            # if self.axis in (decl.name for decl in node.init.decls):
            if node.init.decls[0].name == loop_index:
                memory_space = self.get_memory_space(pragma)
                # Create a new variable name for the cache.
                cache_var = buffer_var + "_" + memory_space.lower()
                # Replace the original buffer variable with the cache variable in the loop body.
                node.stmt.block_items[0].lvalue.name.name = cache_var
                # Create a loop to copy data back from the cache to the original buffer.
                lvalue = c_ast.ArrayRef(
                    name=c_ast.ID(name=buffer_var),
                    subscript=buffer_store_node[0].subscript,
                )
                rvalue = c_ast.ArrayRef(
                    name=c_ast.ID(name=cache_var),
                    subscript=buffer_store_node[0].subscript,
                )
                write_stmt = c_ast.Assignment(op="=", lvalue=lvalue, rvalue=rvalue)
                write_stage_node = c_ast.For(
                    init=node.init,
                    cond=node.cond,
                    next=node.next,
                    stmt=c_ast.Compound([write_stmt]),
                )

                return [node, write_stage_node]
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
    """
    A visitor for extracting pragma information from a C AST.

    Attributes:
    - pragma_info (dict): A dictionary to store the next statement after each pragma.
    - args (dict): A dictionary to store arguments related to the pragma directives.
    """

    def __init__(self):
        """
        Initialize the PragmaVisitor with empty dictionaries for pragma info and args.
        """
        self.pragma_info = {}
        self.args = {}

    def visit_Compound(self, node):
        """
        Visit a Compound node to process pragma directives within it.

        Parameters:
        - node: The AST node for the compound statement (a block of code).
        """
        # Get the block_items, which is a list of statements within the compound statement
        blocks = node.block_items
        for index, node in enumerate(blocks):
            if isinstance(node, c_ast.Pragma):
                self.pragma_info[blocks[index + 1]] = node.string
                stage_visitor = StageVisitor()
                stage_visitor.visit(blocks[index + 1])
                self.args[node.string] = stage_visitor.write_args


class StageVisitor(c_ast.NodeVisitor):
    """
    A visitor for collecting write arguments from assignment statements in a C AST.

    Attributes:
    - write_args (list): A list to store the left-hand side (lvalue) of assignment statements.
    """

    def __init__(self):
        """
        Initialize the StageVisitor with an empty list for write arguments.
        """
        self.write_args = []

    def visit_Assignment(self, node):
        """
        Visit an Assignment node to collect the left-hand side of the assignment.

        Parameters:
        - node: The AST node for the assignment statement being visited.

        This method is called when an assignment statement is encountered in the AST.
        It appends the left-hand side of the assignment (the part before the '=') to the
        write_args list.
        """
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
    cache_write_transform = CacheWriteTransformer(visitor.pragma_info, visitor.args)
    cache_write_transform.visit(ast)
    generator = c_generator.CGenerator()
    print(generator.visit(ast))
