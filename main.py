import ast
import time
import numpy as np
import pandas as pd
import random
import sys
from copy import deepcopy
import argparse
def evaluate(expr, n, x):
    if type(expr) == float or type(expr) == int:
        return expr
    op = expr[0]
    if op == 'add':
        try:
            return evaluate(expr[1], n, x) + evaluate(expr[2], n, x)
        except:

            return random.random()*10
    elif op == 'sub':
        try:
            return evaluate(expr[1], n, x) - evaluate(expr[2], n, x)
        except:
            return random.random()*10
    elif op == 'mul':
        try:
            return evaluate(expr[1], n, x) * evaluate(expr[2], n, x)
        except:
            return sys.maxsize*0.00000001

    # Division operation
    elif op == 'div':
        denominator = evaluate(expr[2], n, x)
        try:
            return evaluate(expr[1], n, x) / denominator if denominator != 0 else 0
        except:
            return random.random() * 10
    elif op == 'pow':
        try:
            base = evaluate(expr[1], n, x)
            exponent = evaluate(expr[2], n, x)

            if abs(exponent) > 100:
                exponent = 100 if exponent > 0 else -100
            if base == 0.0 and exponent < 0:
                return 0
            elif base < 0 and exponent % 1 != 0:
                return 0
            else:
                return base ** exponent
        except OverflowError:
            return sys.maxsize * 0.000001
    elif op == 'sqrt':
        try:
            val = evaluate(expr[1], n, x)
            val = float(val)
            return np.sqrt(val) if val > 0 else 0
        except OverflowError:
            return np.sqrt(sys.maxsize*0.0000001)
        except ValueError:
            return 0
    elif op == 'log':
        try:
            val = evaluate(expr[1], n, x)
            val = float(val)
            return np.log2(val) if val > 0 else 0
        except OverflowError:
            return np.log2(sys.maxsize*0.0000001)
        except:
            return 0
    elif op == 'exp':
        try:
            eval_result = evaluate(expr[1], n, x)
            if eval_result > 700:
                return sys.maxsize * 0.000001
            return np.exp(float(eval_result))
        except OverflowError:
            return sys.maxsize*0.000001
    elif op == 'max':
        return max(evaluate(expr[1], n, x), evaluate(expr[2], n, x))
    elif op == 'ifleq':
        return evaluate(expr[3], n, x) if evaluate(expr[1], n, x) <= evaluate(expr[2], n, x) else evaluate(expr[4], n,
                                                                                                           x)
    elif op == 'data':
        if n <= 0:
            return 0
        index = int(abs(evaluate(expr[1], n, x))) % n
        return x[index]
    elif op == 'diff':
        if n <= 0:
            return 0
        eval_1 = abs(evaluate(expr[1], n, x))
        eval_2 = abs(evaluate(expr[2], n, x))
        try:
            if eval_1 == float('inf'):
                k = n
            elif eval_1 == float('-inf'):
                k = 0
            else:
                k = int(abs(eval_1)) % n
            if eval_2 == float('inf'):
                l = n
            elif eval_2 == float('-inf'):
                l = 0
            else:
                l = int(abs(eval_2)) % n
            return x[k] - x[l]
        except:
            return random.random() * n
    elif op == 'avg':
        if n <= 0:
            return 0
        eval_1 = evaluate(expr[1], n, x)
        eval_2 = evaluate(expr[2], n, x)
        try:
            if eval_1 == float('inf'):
                k = n
            elif eval_1 == float('-inf'):
                k = 0
            else:
                k = int(abs(eval_1)) % n
            if eval_2 == float('inf'):
                l = n
            elif eval_2 == float('-inf'):
                l = 0
            else:
                l = int(abs(eval_2)) % n
            min_index, max_index = min(k, l), max(k, l)
            min_index = max(0, min(n - 1, min_index))
            max_index = max(0, min(n - 1, max_index))
            if min_index == max_index:
                return x[min_index]
            else:
                return np.mean(x[min_index:max_index + 1])
        except:
            return 0
    else:
        return 0
def parse_expression_custom(expr_str):
    def tokenize(s):
        return s.replace('(', ' ( ').replace(')', ' ) ').split()

    def read_from_tokens(tokens):
        if len(tokens) == 0:
            raise SyntaxError('unexpected EOF')
        token = tokens.pop(0)
        if token == '(':
            L = []
            while tokens[0] != ')':
                L.append(read_from_tokens(tokens))
            tokens.pop(0)  # pop off ')'
            return L
        elif token == ')':
            raise SyntaxError('unexpected )')
        else:
            try:
                return ast.literal_eval(token)
            except ValueError:
                return token

    tokens = tokenize(expr_str)
    # print(tokens)
    return read_from_tokens(tokens)

def unparse_expression_custom(expr_list):
    def write_to_string(expr):
        if isinstance(expr, list):
            return '(' + ' '.join(write_to_string(e) for e in expr) + ')'
        else:
            return str(expr)

    return write_to_string(expr_list)


def q1(expr, n, x):
    print(f"Running q 1")
    expr_str = expr
    exprr = parse_expression_custom(expr_str)
    ans = evaluate(exprr, n, x)
    print(f"Answer :{ans}")
    return ans

def q2(expr, n, m, data='expression_data.txt'):
    print("Running q 2")
    file_path = data
    database = pd.read_csv(file_path, sep='\t', header=None)
    X = database.iloc[:, :-1].values
    Y = database.iloc[:, -1].values
    predicted_Y = np.array([q1(expr, n, x) for x in X])
    mse = np.mean((Y - predicted_Y) ** 2)
    return mse

class GeneticProgramming:
    def __init__(self, Lambda, n, m, data_file, time_budget=120, depth=2, tree_type_ratio = 0.8, crossover_rate= 0.5, mutation_rate=0.5, elitism_count = 2):
        self.lambda_population_size = Lambda
        self.n = n
        self.m = m
        self.data_file = data_file
        self.time_budget = time_budget
        self.tree_type_ratio = tree_type_ratio
        self.depth = depth
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_count = elitism_count  # Number of elites

        self.population = []

        self.function_set = {
        'add': 2, 'sub': 2, 'mul': 2, 'div': 2, 'pow': 2, 'sqrt': 1,
        'log': 1, 'exp': 1, 'max': 2, 'ifleq': 4, 'diff': 2, 'avg': 2
    }

        self.terminal_set = ['data']

        self.load_data()
    def generate_non_perfect_tree_as_string(self, depth,  max_depth):
        if depth == max_depth-1:
            return f"(data {random.randint(0, self.n - 1)})"
        else:
            if random.random() > 0.5 or depth == 0:
                func_name, arity = random.choice(list(self.function_set.items()))
                children = [self.generate_non_perfect_tree_as_string(depth + 1, max_depth) for _ in range(arity)]
                children_str = ' '.join(children)
                return f"({func_name} {children_str})"
            else:
                 return f"(data {random.randint(0, self.n - 1)})"


    def generate_perfect_tree_as_string(self, depth, max_depth):
        if depth == max_depth-1:
            return f"(data {random.randint(0, self.n - 1)})"

        else:
            func_name, arity = random.choice([(k, v) for k, v in self.function_set.items() if v != 1 or depth == 0])
            children = [self.generate_perfect_tree_as_string(depth + 1, max_depth) for _ in range(arity)]
            children_str = ' '.join(children)
            return f"({func_name} {children_str})"


    def find_tree_depth(self, tree_string):
        max_depth = 0
        current_depth = 0

        for char in tree_string:
            if char == '(':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ')':
                current_depth -= 1
        return max_depth

    def initialize_population(self):

        num_perfect_trees = int(self.lambda_population_size * self.tree_type_ratio)
        num_non_perfect_trees = self.lambda_population_size - num_perfect_trees
        perfect_trees = [parse_expression_custom(self.generate_perfect_tree_as_string(0, self.depth)) for _ in range(num_perfect_trees)]
        non_perfect_trees = [parse_expression_custom(self.generate_non_perfect_tree_as_string(0, self.depth)) for _ in range(num_non_perfect_trees)]
        self.population = perfect_trees + non_perfect_trees
        random.shuffle(self.population)

    def load_data(self):
        database = pd.read_csv(self.data_file, sep='\t', header=None)
        self.X = database.iloc[:, :-1].values
        self.Y = database.iloc[:, -1].values


    def evaluate_expression_fitness(self, expression):
        predicted_Y = [evaluate(expression, self.n, x) for x in self.X]

        if any(np.isnan(predicted_Y)):
            mse = float('inf')
        else:
            mse = np.mean((self.Y - predicted_Y) ** 2)

        if mse == 0:
            fitness = float('1e10')
        elif np.isinf(mse):
            fitness = 0.0
        else:
            fitness = 1 / (mse + 1e-6)

        return fitness

    def select_parents(self, fitnesses):
        fitnesses = [f if not np.isnan(f) else 0 for f in fitnesses]

        total_fitness = sum(fitnesses)
        probabilities = [fitness / total_fitness for fitness in fitnesses]
        parents_indices = np.random.choice(self.lambda_population_size, size=1, replace=False, p=probabilities)

        parent1 = self.population[parents_indices[0]]
        available_indices = list(range(self.lambda_population_size))
        available_indices.remove(parents_indices[0])
        parent2_index = np.random.choice(available_indices)
        parent2 = self.population[parent2_index]

        return parent1, parent2

    def evaluate_expression_fitness(self, expression):
        try:
            predicted_Y = [evaluate(expression, self.n, x) for x in self.X]
        except Exception as e:
            return 1e-15
        diff = self.Y - np.array(predicted_Y)
        max_diff = 1e18
        diff_clipped = np.clip(diff, -max_diff, max_diff)
        mse = np.mean(diff_clipped ** 2)
        if np.isnan(mse) or np.isinf(mse) :
            return 1e-15
        if mse == 0:
            return 1e18
        else:
            return 1 / (mse)


    def crossover(self, parent1, parent2):
        def select_crossover_point(tree):
            def dfs(node, path, indices):
                if isinstance(node, list):
                    if path:
                        indices.append(path.copy())
                    for i, child in enumerate(node):
                        dfs(child, path + [i], indices)
            indices = []
            dfs(tree, [], indices)
            return random.choice(indices) if indices else None

        def get_subtree_at_path(tree, path):
            for index in path:
                tree = tree[index]
            return tree

        def set_subtree_at_path(tree, path, new_subtree):
            if path:
                target = tree
                for index in path[:-1]:
                    target = target[index]
                target[path[-1]] = new_subtree
            else:
                tree[:] = new_subtree

        crossover_point1 = select_crossover_point(parent1)
        crossover_point2 = select_crossover_point(parent2)
        if crossover_point1 is None or crossover_point2 is None:
            return parent1, parent2
        subtree1 = get_subtree_at_path(parent1, crossover_point1)
        subtree2 = get_subtree_at_path(parent2, crossover_point2)
        offspring1 = deepcopy(parent1)
        offspring2 = deepcopy(parent2)
        set_subtree_at_path(offspring1, crossover_point1, subtree2)
        set_subtree_at_path(offspring2, crossover_point2, subtree1)
        return offspring1 ,offspring2

    def mutate(self, expression):
        function_nodes = []
        def find_functions(node, path=[]):
            if isinstance(node, list) and node:
                if node[0] in self.function_set:
                    function_nodes.append((path.copy(), node[0]))
                for i, child in enumerate(node[1:], start=1):
                    find_functions(child, path + [i])
        find_functions(expression)
        if not function_nodes:
            return expression
        selected_path, selected_function = random.choice(function_nodes)
        def replace_function(node, path, replacement_function):
            if not path:
                node[0] = replacement_function
            else:
                for step in path[:-1]:
                    node = node[step]
                node[path[-1]][0] = replacement_function
        arity = self.function_set[selected_function]
        possible_replacements = [f for f, a in self.function_set.items() if a == arity and f != selected_function]
        x = deepcopy(expression)
        if possible_replacements:
            replacement_function = random.choice(possible_replacements)
            replace_function(x, selected_path, replacement_function)
        return x

    def calculate_tree_depth(self,tree):
        if not isinstance(tree, list) or not tree:
            return 0
        else:
            return 1 + max(self.calculate_tree_depth(subtree) for subtree in tree)

    def prune_tree(self, tree, max_depth, current_depth=0):
        if current_depth == max_depth - 1:
            for i, subtree in enumerate(tree):
                if isinstance(subtree, list):
                    x = random.randint(0, self.n - 1)
                    tree[i] = ['data', x]
        else:
            for i, subtree in enumerate(tree):
                if isinstance(subtree, list):
                    self.prune_tree(subtree, max_depth, current_depth + 1)
        return tree

    def run(self):
        print("Running q 3")
        start_time = time.time()
        self.initialize_population()
        self.fitnesses = [self.evaluate_expression_fitness(expr) for expr in self.population]
        last_print_time = start_time
        generation_counter = 0
        fitnesses_over_generations = []
        best_global_fitness = -np.inf
        best_global_expression = None

        while time.time() - start_time < self.time_budget:
            if generation_counter % 50 == 0 or generation_counter ==0:
                sum_fitness = sum(self.fitnesses)
                best_index = np.argmax(self.fitnesses)
                # print(f"Generation: {generation_counter}, Best Fitness: {self.fitnesses[best_index]}, Total Fitness: {sum(self.fitnesses)}, Best Expr: {self.population[best_index]} ")
                fitnesses_over_generations.append(sum_fitness)
            generation_counter += 1
            new_population = []
            elites = np.argsort(self.fitnesses)[-self.elitism_count:]
            for idx in elites:
                new_population.append(self.population[idx])


            while len(new_population) < len(self.population):
                parent1, parent2 = self.select_parents(self.fitnesses)
                if random.random() < self.crossover_rate:
                    offspring1, offspring2 = self.crossover(parent1, parent2)
                    offspring2 = self.prune_tree(offspring2,self.depth,0)
                    offspring1 = self.prune_tree(offspring1,self.depth,0)
                else:
                    offspring1 = parent1
                    offspring2 = parent2
                if random.random() < self.mutation_rate:
                    offspring1 = self.mutate(offspring1)
                new_population.append(offspring1)
                new_population.append(offspring2)
                if len(new_population) > len(self.population):
                    new_population.pop()

            self.population = new_population
            self.fitnesses = [self.evaluate_expression_fitness(expr) for expr in self.population]

            current_best_fitness = max(self.fitnesses)
            current_best_index = self.fitnesses.index(current_best_fitness)
            current_best_expression = self.population[current_best_index]

            if current_best_fitness > best_global_fitness:
                best_global_fitness = current_best_fitness
                best_global_expression = current_best_expression


            if time.time() - last_print_time >= 30:
                # print(f"Time elapsed: {time.time() - start_time:.2f}s, Best Fitness: {best_global_fitness}, Best Expr: {best_global_expression}")
                last_print_time = time.time()
        expr_str = unparse_expression_custom(best_global_expression)
        return expr_str

def q3(Lambda , n, m, data_file= 'addition_2', time_budget=60*10, depth=2, tree_type_ratio =0.2, crossover_rate = 0.7, mutation_rate = 0.7,elitism_count = 10):
    print(f'Starting q3')
    gp = GeneticProgramming(Lambda, n, m, data_file, time_budget, depth, tree_type_ratio, crossover_rate, mutation_rate, elitism_count)
    # print('Initialised the GP class')
    best_expression = gp.run()
    return best_expression




if __name__ == '__main__':


    parser = argparse.ArgumentParser(description="Lab Solution")
    parser.add_argument("--q", type=int, required=False, help="q number: 1, 2, or 3")
    parser.add_argument("--Lambda", type=int, default=100, help="Population size (default: 100)")

    parser.add_argument("--n", type=int, required=False, help="Input n (required)")
    parser.add_argument("--m", type=int, required=False, help="Input m (required)")

    parser.add_argument("--x", nargs='+', type=float, help="Input x (multiple values)")
    parser.add_argument("--data", type=str, help="Data file name")
    parser.add_argument("--time_budget", type=int, help="Time budget in seconds")

    parser.add_argument("--expr", type=str, required=True, help="Expression to evaluate (required)")

    args = parser.parse_args()

    if args.q == 1:
        if not args.x:
            parser.error("q 1 requires -x arguments.")
        q1(args.expr, args.n, args.x)
    elif args.q == 2:
        if not args.data:
            parser.error("q 2 requires -data argument.")
        q2(args.expr, args.n, args.m, args.data)
    elif args.q == 3:
        if not all([args.Lambda, args.data, args.time_budget]):
            parser.error("q 3 requires -Lambda, -data, and -time_budget arguments.")
        depth = args.n % 3 + 3
        q3(args.Lambda, args.n, args.m, args.data, args.time_budget, depth)
    else:
        print("Invalid q number")