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
    return modified_code
