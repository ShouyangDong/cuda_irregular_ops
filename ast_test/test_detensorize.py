from pycparser import c_parser, c_ast, parse_file


# A visitor with some state information (the funcname it's
# looking for)
#
class FuncCallVisitor(c_ast.NodeVisitor):
    def __init__(self, funcname):
        self.funcname = funcname

    def visit_FuncCall(self, node):
        if node.name.name == self.funcname:
            print("%s called at %s" % (self.funcname, node.name.coord))


def show_func_calls(filename, funcname):
    ast = parse_file(filename, use_cpp=True)
    v = FuncCallVisitor(funcname)
    v.visit(ast)


if __name__ == "__main__":
    parser = c_parser.CParser()
    ast = parser.parse(text)
    # print("AST before change:")
    # ast.show(offset=2)

    v = ParamAdder()
    v.visit(ast)

    # print("\nAST after change:")
    # ast.show(offset=2)

    print("\nCode after change:")
    generator = c_generator.CGenerator()
    print(generator.visit(ast))
