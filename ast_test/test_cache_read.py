from pycparser import c_parser, c_ast, c_generator


class NodeTransformer(c_ast.NodeVisitor):
    """
    A node transformer that visits each node in an AST and applies transformations.

    Attributes:
    - None explicitly defined here, but subclasses may add attributes.
    """

    def generic_visit(self, node):
        """
        A generic visit method that is called for nodes that don't have a specific visit_<nodetype> method.

        This method iterates over all fields in the current node. If a field contains a list of nodes,
        it applies the transformation to each item in the list. If a field contains a single node, it applies
        the transformation to that node.

        Parameters:
        - node: The AST node to visit and potentially transform.

        Returns:
        - The original node, potentially with some of its fields transformed or replaced.
        """
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
    """
    Iterate over all fields of a pycparser AST node.

    Parameters:
    - node: The AST node whose fields are to be iterated over.

    Yields:
    - A tuple containing the name of the field and the value of the field.
    """
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


class CacheReadTransformer(NodeTransformer):
    """
    A transformer that modifies loops to perform cache reads based on pragma directives.

    Attributes:
    - pragma_info (dict): A dictionary mapping nodes to pragma strings.
    - args (dict): A dictionary mapping buffer names to pragma strings.
    """

    def __init__(self, pragma_info, args):
        self.pragma_info = pragma_info
        self.args = args
        self.node_substitude_dict = {}

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
        Transform the for loop to perform cache reads as specified by the pragma.

        Parameters:
        - node: The AST node for the for loop.

        Returns:
        - A list of new AST nodes that replace the original for loop.
        """

        def are_nodes_equal(node1, node2):
            """
            Compare two AST nodes for equality.

            Parameters:
            - node1: The first AST node.
            - node2: The second AST node.

            Returns:
            - True if the nodes are equal, False otherwise.
            """
            # Check if both nodes are of the same type
            if type(node1) != type(node2):
                return False
            # TODO: please add the node check.
            return True

        def replace_id_name(node):
            if isinstance(node, c_ast.ID) and node.name in self.node_substitude_dict:
                node.name = self.node_substitude_dict[node.name]
            for field, value in iter_fields(node):
                if isinstance(value, list):
                    for item in value:
                        replace_id_name(item)
                elif isinstance(value, c_ast.Node):
                    replace_id_name(value)

        # TODO: maybe use list instead of dictionary
        for pragma, buffer_store_node in self.args.items():
            # Check two node are equal
            if not are_nodes_equal(node, self.pragma_info[pragma]):
                return node

            else:
                stmt_list = []
                # Traverse the buffer list
                for index, buffer_node in enumerate(buffer_store_node):
                    buffer_var = buffer_node.name.name
                    loop_index = buffer_node.subscript.name
                    memory_space = self.get_memory_space(pragma, index)
                    # Create a new variable name for the cache.
                    cache_var = buffer_var + "_" + memory_space.lower()
                    self.node_substitude_dict[buffer_var] = cache_var
                    # Create a loop to copy data back from the cache to the original buffer.
                    lvalue = c_ast.ArrayRef(
                        name=c_ast.ID(name=cache_var),
                        subscript=buffer_node.subscript,
                    )
                    rvalue = c_ast.ArrayRef(
                        name=c_ast.ID(name=buffer_var),
                        subscript=buffer_node.subscript,
                    )
                    read_stmt = c_ast.Assignment(op="=", lvalue=lvalue, rvalue=rvalue)

                    read_stage_node = c_ast.For(
                        init=node.init,
                        cond=node.cond,
                        next=node.next,
                        stmt=c_ast.Compound([read_stmt]),
                    )
                    stmt_list.append(read_stage_node)

                # Replace the original buffer variable with the cache variable in the loop body.
                replace_id_name(node)
                stmt_list.append(node)
                return stmt_list

    def get_memory_space(self, pragma, index):
        """
        Extract the memory space from the pragma string.

        Parameters:
        - pragma (str): The pragma string containing memory space information.

        - index (int): The index of input argument.
        Returns:
        - The extracted memory space as a string.
        """
        return pragma.split("input[")[1].split("]")[0].split(",")[index].strip()


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
                self.pragma_info[node.string] = blocks[index + 1]
                stage_visitor = StageVisitor()
                stage_visitor.visit(blocks[index + 1])
                self.args[node.string] = stage_visitor.read_args


class StageVisitor(c_ast.NodeVisitor):
    """
    A visitor for collecting read arguments from assignment statements in a C AST.

    Attributes:
    - read_args (list): A list to store the right-hand side (lvalue) of assignment statements.
    """

    def __init__(self):
        """
        Initialize the StageVisitor with an empty list for read arguments.
        """
        self.read_args = []

    def visit_Assignment(self, node):
        """
        Visit an Assignment node to collect the right-hand side of the assignment.

        Parameters:
        - node: The AST node for the assignment statement being visited.

        This method is called when an assignment statement is encountered in the AST.
        It appends the right-hand side of the assignment (the part before the '=') to the
        read_args list.
        """

        if isinstance(node.rvalue, c_ast.BinaryOp):
            self.read_args.append(node.rvalue.left)
            self.read_args.append(node.rvalue.right)

        elif isinstance(ndoe.rvalue, c_ast.UnaryOp):
            # FIXME: The key is incorrect.
            self.read_args.append(node.rvalue)

        else:
            raise RuntimeError("Cannot not handle this assignment")


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
    cache_read_transform = CacheReadTransformer(visitor.pragma_info, visitor.args)
    cache_read_transform.visit(ast)
    generator = c_generator.CGenerator()
    print(generator.visit(ast))
