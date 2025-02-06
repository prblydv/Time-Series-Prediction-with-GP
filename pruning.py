import random


def calculate_tree_depth(tree):
    """
    Recursively calculates the depth of a tree represented as nested lists.
    """
    if not isinstance(tree, list) or not tree:
        return 0
    else:
        return 1 + max(calculate_tree_depth(subtree) for subtree in tree)

def prune_tree(tree, max_depth, current_depth=0):
    """
    Prunes the tree if its depth exceeds max_depth. This implementation
    removes nodes or replaces subtrees with leaf nodes (e.g., a terminal node)
    to ensure the tree depth does not exceed max_depth.
    """
    # If the current depth is at max_depth - 1, replace any subtrees with a leaf node
    if current_depth == max_depth - 1:
        for i, subtree in enumerate(tree):
            if isinstance(subtree, list):
                # Replace subtree with a terminal node, adjust as needed
                x =random.randint(0, 5 - 1)
                tree[i] = ['data', x]  # Example terminal node
    else:
        for i, subtree in enumerate(tree):
            if isinstance(subtree, list):
                prune_tree(subtree, max_depth, current_depth + 1)
    return tree

# Example usage
if __name__ == "__main__":

    max_depth = 1  # Maximum allowed depth
    parent1 = ['ifleq', ['diff', ['sub', ['data', 1], ['data', 3]], ['data', 3]], ['data', 2], ['max', ['mul', ['pow', ['data', 1], ['data', 1]], ['max', ['data', 1], ['data', 2]]], ['data', 2]], ['data', 2]]
    print("Parent tree:", parent1)
    print("Original tree depth:", calculate_tree_depth(parent1))
    pruned_tree = prune_tree(parent1, max_depth)

    print("Pruned tree:", pruned_tree)
    print("Pruned tree depth:", calculate_tree_depth(pruned_tree))
