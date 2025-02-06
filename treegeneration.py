import random
from main import parse_expression_custom, evaluate

def find_tree_depth(tree):
    if not isinstance(tree, list) or not tree:
        # Base case: if the tree is not a list or is an empty list, its depth is 0
        return 0
    else:
        # Recursively find the depth of each subtree
        depths = [find_tree_depth(subtree) for subtree in tree if isinstance(subtree, list)]
        # If there are no sublists (meaning no deeper subtrees), return 1 (the depth of the current node)
        if not depths:
            return 1
        else:
            # Return 1 plus the maximum depth among the subtrees
            return 1 + max(depths)


def unparse_expression_custom(expr_list):
    def write_to_string(expr):
        if isinstance(expr, list):
            return '(' + ' '.join(write_to_string(e) for e in expr) + ')'
        else:
            return str(expr)

    return write_to_string(expr_list)

# Example usage
expr_list = ['add', ['mul', ['data', 3], 3], ['sub', 4, ['data', 0]]]
expr_str = unparse_expression_custom(expr_list)
print(expr_str)


n=4
function_set = {
        'add': 2, 'sub': 2, 'mul': 2, 'div': 2, 'pow': 2, 'sqrt': 1,
        'log': 1, 'exp': 1, 'max': 2, 'ifleq': 4, 'diff': 2, 'avg': 2
    }
def generate_non_perfect_tree_as_string( depth, max_depth):
    if depth == max_depth-1:
        # At maximum depth, choose only from the terminal set
        # return f"(data {random.randint(0, self.n - 1)})"
        # Return a 'data' terminal
        return f"(data {random.randint(0, n - 1)})"
    else:
        # Select randomly between function set and terminal set
        if random.random() > 0.5 or depth == 0:  # Ensure not to pick terminal at root
            # Pick a function and recursively build its children
            func_name, arity = random.choice(list(function_set.items()))
            children = [generate_non_perfect_tree_as_string(depth + 1, max_depth) for _ in range(arity)]
            children_str = ' '.join(children)
            return f"({func_name} {children_str})"
        else:
            return f"(data {random.randint(0, n - 1)})"


def generate_perfect_tree_as_string( depth, max_depth):
    if depth == max_depth-1:
        # At the leaf level, choose only from the terminal set
        # return f"(data {random.randint(0, self.n - 1)})"
        return f"(data {random.randint(0, n - 1)})"

    else:
        # At non-leaf levels, ensure to pick a function and expand fully according to its arity
        func_name, arity = random.choice([(k, v) for k, v in function_set.items() if v != 1 or depth == 0])
        # Recursively build child nodes for each function to maintain the perfect tree structure
        children = [generate_perfect_tree_as_string(depth + 1, max_depth) for _ in range(arity)]
        children_str = ' '.join(children)
        return f"({func_name} {children_str})"



if __name__ == '__main__':
    tree = generate_perfect_tree_as_string(0,1)
    print(parse_expression_custom(tree))
    print(find_tree_depth(['mul', ['data', 0], ['ifleq', ['data', 2], ['data', 2], ['ifleq', ['data', 3], ['ifleq', ['ifleq', ['data', 2], ['data', 0], ['data', 2], ['data', 2]], ['data', 2], ['data', 2], ['data', 2]], ['ifleq', ['data', 3], ['sqrt', ['data', 0]], ['ifleq', ['data', 2], ['ifleq', ['data', 2], ['data', 1], ['ifleq', ['log', ['data', 3]], ['data', 2], ['ifleq', ['data', 2], ['data', 2], ['data', 0], ['data', 2]], ['data', 2]], ['data', 2]], ['ifleq', ['data', 2], ['data', 2], ['data', 2], ['data', 2]], ['ifleq', ['data', 2], ['data', 2], ['ifleq', ['data', 0], ['ifleq', ['data', 2], ['data', 2], ['data', 0], ['data', 2]], ['data', 2], ['ifleq', ['data', 2], ['data', 1], ['data', 2], ['data', 2]]], ['data', 2]]], ['ifleq', ['data', 2], ['data', 1], ['data', 3], ['data', 2]]], ['data', 2]], ['ifleq', ['data', 0], ['data', 2], ['ifleq', ['data', 2], ['data', 1], ['data', 2], ['data', 1]], ['ifleq', ['data', 0], ['data', 2], ['data', 2], ['ifleq', ['data', 2], ['data', 1], ['ifleq', ['data', 2], ['data', 3], ['data', 2], ['data', 2]], ['ifleq', ['data', 0], ['data', 0], ['data', 2], ['data', 2]]]]]]]
))
    print(unparse_expression_custom(['mul', ['data', 0], ['ifleq', ['data', 2], ['data', 2], ['ifleq', ['data', 3], ['ifleq', ['ifleq', ['data', 2], ['data', 0], ['data', 2], ['data', 2]], ['data', 2], ['data', 2], ['data', 2]], ['ifleq', ['data', 3], ['sqrt', ['data', 0]], ['ifleq', ['data', 2], ['ifleq', ['data', 2], ['data', 1], ['ifleq', ['log', ['data', 3]], ['data', 2], ['ifleq', ['data', 2], ['data', 2], ['data', 0], ['data', 2]], ['data', 2]], ['data', 2]], ['ifleq', ['data', 2], ['data', 2], ['data', 2], ['data', 2]], ['ifleq', ['data', 2], ['data', 2], ['ifleq', ['data', 0], ['ifleq', ['data', 2], ['data', 2], ['data', 0], ['data', 2]], ['data', 2], ['ifleq', ['data', 2], ['data', 1], ['data', 2], ['data', 2]]], ['data', 2]]], ['ifleq', ['data', 2], ['data', 1], ['data', 3], ['data', 2]]], ['data', 2]], ['ifleq', ['data', 0], ['data', 2], ['ifleq', ['data', 2], ['data', 1], ['data', 2], ['data', 1]], ['ifleq', ['data', 0], ['data', 2], ['data', 2], ['ifleq', ['data', 2], ['data', 1], ['ifleq', ['data', 2], ['data', 3], ['data', 2], ['data', 2]], ['ifleq', ['data', 0], ['data', 0], ['data', 2], ['data', 2]]]]]]]))
    print(evaluate(['add', ['mul', ['data', 3], 3], ['sub', 4, ['data', 0]]], 4 ,[5,1,2,3,4]))
    print('\n')

    tree2 = generate_non_perfect_tree_as_string(0,5)
    print(parse_expression_custom(tree2))
    print(find_tree_depth(tree2))

