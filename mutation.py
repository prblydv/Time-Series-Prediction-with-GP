import random
from main import question1, evaluate
from copy import deepcopy


function_set = {
    'add': 2, 'sub': 2, 'mul': 2, 'div': 2, 'pow': 2, 'sqrt': 1,
    'log': 1, 'exp': 1, 'max': 2, 'ifleq': 4, 'diff': 2, 'avg': 2
}


def mutate(expression):
    function_nodes = []

    def find_functions(node, path=[]):
        if isinstance(node, list) and node:  # Check if it's a non-empty list (a node in the tree)
            if node[0] in function_set:  # It's a function node
                function_nodes.append((path.copy(), node[0]))  # Record the path and function name
            for i, child in enumerate(node[1:], start=1):  # Iterate children
                find_functions(child, path + [i])  # Recursive search

    find_functions(expression)

    if not function_nodes:
        return expression  # No function nodes to mutate

    # Select a random function node to mutate
    print('1',function_nodes)
    selected_path, selected_function = random.choice(function_nodes)
    print('2 ',selected_path)
    print('3 ',selected_function)


    # Function to replace the selected function node with a mutated version
    def replace_function(node, path, replacement_function):

        if not path:  # If path is empty, mutate the root node
            node[0] = replacement_function
        else:
            for step in path[:-1]:
                node = node[step]  # Navigate to the parent of the target node
            node[path[-1]][0] = replacement_function  # Mutate the function

        # for step in path[:-1]:
        #     node = node[step]  # Navigate to the parent of the target node
        # # Mutate the function
        # if path:  # Check if path is not empty
        #     node[path[-1]][0] = replacement_function

    # Choose a replacement function with the same arity
    arity = function_set[selected_function]
    possible_replacements = [f for f, a in function_set.items() if a == arity and f != selected_function]
    x = deepcopy(expression)
    if possible_replacements:
        print('4 ', possible_replacements)
        replacement_function = random.choice(possible_replacements)
        print('5 ', replacement_function)

        # x =
        replace_function(x, selected_path, replacement_function)

    return x




if __name__ == '__main__':
    o1, o2 = ['exp', ['max', ['data', 0], ['data', 1]]], ['mul', ['max', ['data', 2], ['data', 3]], ['div', ['data', 4], ['data', 5]]]
    mutated = mutate(o1)
    print(evaluate(o1, 5, [0,1,2,3,4,5]))
    print(mutated)
    print(evaluate(mutated, 5, [0,1,2,3,4,5]))
