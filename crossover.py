import random
# def crossover(parent1, parent2):
#     def select_crossover_point(tree):
#         def dfs(node, path, indices):
#             if isinstance(node, list):
#                 indices.append(path.copy())  # Add current path as potential crossover point
#                 for i, child in enumerate(node):
#                     dfs(child, path + [i], indices)  # Recursively search through children
#
#         indices = []
#         dfs(tree, [], indices)  # Initialize DFS with an empty path
#         print('1 ',indices)
#         return random.choice(indices) if indices else None
#
#     def get_subtree_at_path(tree, path):
#         for index in path:
#             tree = tree[index]
#         print('2 ',tree)
#
#         return tree
#
#     def set_subtree_at_path(tree, path, new_subtree):
#         target = tree
#         for index in path[:-1]:
#             target = target[index]
#         target[path[-1]] = new_subtree
#
#     # Select random crossover points in both parents
#     crossover_point1 = select_crossover_point(parent1)
#     print('3', crossover_point1)
#     crossover_point2 = select_crossover_point(parent2)
#     print('4', crossover_point2)
#     if crossover_point1 is None or crossover_point2 is None:
#         # If no valid crossover point is found in either parent, return the parents unchanged
#         return parent1, parent2
#
#     # Extract subtrees at the selected crossover points
#     subtree1 = get_subtree_at_path(parent1, [0])
#     print('5 ', subtree1)
#
#     subtree2 = get_subtree_at_path(parent2, [2])
#     print('6 ', subtree2)
#
#     # Create copies of the parents to avoid altering the original parents
#     offspring1 = parent1.copy()
#     offspring2 = parent2.copy()
#
#     # Swap the subtrees at the crossover points
#     set_subtree_at_path(offspring1, crossover_point1, subtree2)
#     set_subtree_at_path(offspring2, crossover_point2, subtree1)
#
#     return offspring1, offspring2
from treegeneration import generate_perfect_tree_as_string, generate_non_perfect_tree_as_string
from main import parse_expression_custom, evaluate
import random
from copy import deepcopy


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


# def crossover(parent1, parent2):
#     def select_crossover_point(tree):
#         def dfs(node, path, indices):
#             if isinstance(node, list):
#                 if path:  # Exclude the root to avoid replacing the entire tree
#                     indices.append(path.copy())  # Add current path as a potential crossover point
#                 for i, child in enumerate(node):
#                     dfs(child, path + [i], indices)  # Recursively search through children
#
#         indices = []
#         dfs(tree, [], indices)  # Initialize DFS with an empty path
#         return random.choice(indices) if indices else None
#
#     def select_crossover_point_with_depth_limit(tree, max_depth):
#         def dfs(node, path, indices, current_depth):
#             if current_depth > max_depth:
#                 return  # Do not add indices beyond the maximum depth
#             if isinstance(node, list):
#                 if path:  # Exclude the root
#                     indices.append(path.copy())
#                 for i, child in enumerate(node):
#                     dfs(child, path + [i], indices, current_depth + 1)
#
#         indices = []
#         dfs(tree, [], indices, 0)
#         return random.choice(indices) if indices else None
#
#     def get_subtree_at_path(tree, path):
#         for index in path:
#             tree = tree[index]
#         return tree
#
#     def set_subtree_at_path(tree, path, new_subtree):
#         target = tree
#         for index in path[:-1]:
#             target = target[index]
#         target[path[-1]] = new_subtree
#     maxdP1 = find_tree_depth(parent1)
#     maxdP2 = find_tree_depth(parent2)
#
#     depth = min(maxdP1,maxdP2)
#
#     # Select random crossover points in both parents
#     crossover_point1 = select_crossover_point_with_depth_limit(parent1,depth)
#     crossover_point2 = select_crossover_point_with_depth_limit(parent2,depth)
#
#     # If no valid crossover point is found in either parent, return the parents unchanged
#     if crossover_point1 is None or crossover_point2 is None:
#         print('crossover point not found')
#         return parent1, parent2
#
#     # Extract subtrees at the selected crossover points
#     subtree1 = get_subtree_at_path(parent1, crossover_point1)
#     subtree2 = get_subtree_at_path(parent2, crossover_point2)
#
#     # Create deep copies of the parents to avoid altering the original parents
#     offspring1 = deepcopy(parent1)
#     offspring2 = deepcopy(parent2)
#
#     # Swap the subtrees at the crossover points
#     set_subtree_at_path(offspring1, crossover_point1, subtree2)
#     set_subtree_at_path(offspring2, crossover_point2, subtree1)
#
#     return offspring1, offspring2
#
import random
from copy import deepcopy

# def crossover(parent1, parent2):
#     def find_subtrees_by_depth(tree, current_depth=0, path=[], depth_subtree_map={}):
#         if isinstance(tree, list):
#             # Record the subtree at this path with its depth
#             if path:  # Exclude the root to ensure we're looking at actual subtrees
#                 depth_subtree_map.setdefault(current_depth, []).append(path.copy())
#             for i, child in enumerate(tree):
#                 find_subtrees_by_depth(child, current_depth + 1, path + [i], depth_subtree_map)
#         return depth_subtree_map
#
#     def get_subtree_at_path(tree, path):
#         for index in path:
#             if index >= len(tree):
#                 raise ValueError("Path leads to a non-existent index in the tree")
#             tree = tree[index]
#         return tree
#
#     def set_subtree_at_path(tree, path, new_subtree):
#         target = tree
#         for index in path[:-1]:
#             if index >= len(target):
#                 raise ValueError("Path leads to a non-existent index in the tree")
#             target = target[index]
#         if path[-1] >= len(target):
#             raise ValueError("Final step in path leads to a non-existent index in the tree")
#         target[path[-1]] = new_subtree
#
#     depth_subtree_map1 = find_subtrees_by_depth(parent1)
#     print("Depth map",depth_subtree_map1)
#     depth_subtree_map2 = find_subtrees_by_depth(parent2)
#
#     # Find common depths
#     common_depths = set(depth_subtree_map1.keys()) & set(depth_subtree_map2.keys())
#     print('1 ',common_depths)
#     if not common_depths:
#         print('No common depths for crossover.')
#         return parent1, parent2
#
#     # Choose a random common depth
#     chosen_depth = random.choice(list(common_depths))
#
#     # Select random crossover points at the chosen depth
#     crossover_point1 = random.choice(depth_subtree_map1[chosen_depth])
#     crossover_point2 = random.choice(depth_subtree_map2[chosen_depth])
#
#     # Extract subtrees at the selected crossover points
#     subtree1 = get_subtree_at_path(parent1, crossover_point1)
#     subtree2 = get_subtree_at_path(parent2, crossover_point2)
#
#     # Create deep copies of the parents to avoid altering the original parents
#     offspring1 = deepcopy(parent1)
#     offspring2 = deepcopy(parent2)
#
#     # Swap the subtrees at the crossover points
#     set_subtree_at_path(offspring1, crossover_point1, subtree2)
#     set_subtree_at_path(offspring2, crossover_point2, subtree1)
#
#     return offspring1, offspring2
#
#
def crossover(parent1, parent2):
    def find_subtrees_by_depth(tree, current_depth=0, path=[], depth_subtree_map={}):
        if isinstance(tree, list):
            if path:  # Exclude the root
                depth_subtree_map.setdefault(current_depth, []).append(path.copy())
            for i, child in enumerate(tree):
                find_subtrees_by_depth(child, current_depth + 1, path + [i], depth_subtree_map)
        return depth_subtree_map

    def get_subtree_at_path(tree, path):
        for index in path:
            if index >= len(tree):
                return None  # Path invalid
            tree = tree[index]
        return tree

    def set_subtree_at_path(tree, path, new_subtree):
        target = tree
        for index in path[:-1]:
            if index >= len(target):
                raise ValueError("Path leads to a non-existent index in the tree")
            target = target[index]
        if path[-1] >= len(target):
            raise ValueError("Final step in path leads to a non-existent index in the tree")
        target[path[-1]] = new_subtree


    depth_subtree_map1 = find_subtrees_by_depth(parent1)
    depth_subtree_map2 = find_subtrees_by_depth(parent2)

    common_depths = set(depth_subtree_map1.keys()) & set(depth_subtree_map2.keys())
    if not common_depths:
        print('No common depths for crossover.')
        return parent1, parent2, False

    chosen_depth = random.choice(list(common_depths))
    crossover_point1 = random.choice(depth_subtree_map1[chosen_depth])
    crossover_point2 = random.choice(depth_subtree_map2[chosen_depth])

    subtree1 = get_subtree_at_path(parent1, crossover_point1)
    subtree2 = get_subtree_at_path(parent2, crossover_point2)
    if subtree1 is None or subtree2 is None:
        print('Invalid crossover point.')
        return parent1, parent2 , False # Avoid crossover if path is invalid

    offspring1 = deepcopy(parent1)
    offspring2 = deepcopy(parent2)
    set_subtree_at_path(offspring1, crossover_point1, subtree2)
    set_subtree_at_path(offspring2, crossover_point2, subtree1)

    return offspring1, offspring2, True



if __name__ == '__main__':
    n = 6
    same_trees = 0
    no_change = 0
    for i in range(100):
        parent1 = generate_perfect_tree_as_string(0,n)
        parent2 = generate_perfect_tree_as_string(0,n-2)

        parent1 = parse_expression_custom(parent1)
        print('4.', parent1 )
        a =find_tree_depth(parent1)
        a1 =evaluate(parent1, 4, [1,2,3,4])
        print('5.',a, a1)


        parent2 = parse_expression_custom(parent2)
        print('5.5. ',parent2)
        b= find_tree_depth(parent2)
        b1 = evaluate(parent2, 4, [1,2,3,4])
        print('6. ', b,b1)


        o1, o2 , crosshappend = crossover(parent1,parent2)
        print('7. ',o1)
        c = find_tree_depth(o1)
        c1 = evaluate(o1, 4, [1,2,3,4])
        print('8. ', c, c1)
        print('9. ',o2)
        d = find_tree_depth(o2)
        d1 = evaluate(o2, 4, [1,2,3,4])
        print('10. ', d, d1)
        if a==c and b==c and b == d:
            same_trees +=1
        if crosshappend == False:
            no_change += 1
    print( 'total no of same trees depth ',same_trees)
    print( 'total no change in trees ',no_change)


