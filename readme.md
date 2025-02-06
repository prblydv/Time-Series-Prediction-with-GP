# **Genetic Programming for Mathematical Expression Optimization**

## 📌 **Introduction**
This project implements **Genetic Programming (GP)** to optimize mathematical expressions that model relationships between data points. The algorithm evolves expressions to fit given data using **genetic operators** like selection, crossover, and mutation.

### **Key Features**
- ✅ Evaluates mathematical expressions using a tree-based representation.
- ✅ Implements **Genetic Programming (GP)** for automatic function discovery.
- ✅ Supports **Simulated Annealing** for further optimization.
- ✅ Implements **custom mathematical operations** (e.g., power, logarithm, exponential, etc.).
- ✅ Parses and converts **mathematical expressions into tree structures** for easier evolution.
- ✅ Supports **command-line arguments** to allow flexible execution.

---

## **📂 File Structure**
- `main.py` – Main script containing the GP implementation and evaluation methods.
- `expression_data.txt` – Data file used for evaluating expressions.
- `README.md` – Documentation for the project.
- `requirements.txt` – Dependencies for running the project.

---

## **🚀 How It Works**
### **1️⃣ Mathematical Expression Evaluation**
The `evaluate(expr, n, x)` function recursively evaluates mathematical expressions represented as a **tree structure**.

#### **Supported Operations**
| Operation  | Description |
|------------|------------|
| `add` | Addition |
| `sub` | Subtraction |
| `mul` | Multiplication |
| `div` | Division (Handles division by zero) |
| `pow` | Power function (Restricts exponentiation range) |
| `sqrt` | Square root |
| `log` | Logarithm (Handles negative values) |
| `exp` | Exponential function |
| `max` | Maximum of two values |
| `ifleq` | Conditional IF statement (if A ≤ B, then X else Y) |
| `data` | Fetches indexed values from input data |
| `diff` | Computes the difference between two indexed values |
| `avg` | Computes the average of values in a given range |

### **2️⃣ Expression Parsing**
Mathematical expressions are represented as **nested lists**, which function as a **tree structure**.  
For example, the expression:  
```python
(add (mul 3 x) (sub x 5))
```
Is parsed into:
```python
['add', ['mul', 3, 'x'], ['sub', 'x', 5]]
```
- The **parser converts strings into tree structures**.
- The **unparser reconstructs expressions back into readable format**.


---

## **🔢 Genetic Programming (GP) Implementation**
### **3️⃣ Genetic Algorithm for Function Evolution**
The `GeneticProgramming` class implements **tree-based GP** to evolve mathematical expressions for **symbolic regression**.

#### **Steps:**
1. **Initialize Population**  
   - Random trees are generated using **two methods**:
     - **Perfect Trees** (Full depth, balanced).
     - **Non-Perfect Trees** (Irregular depth, mixed complexity).
   - Functions are selected from the predefined function set.

2. **Fitness Evaluation**  
   - Expressions are evaluated on dataset `expression_data.txt` using **Mean Squared Error (MSE)**.
   - Lower MSE means better fitness.

3. **Selection**  
   - **Roulette-wheel selection** is used based on fitness scores.
   - Parents are chosen **proportionally to their fitness**.

4. **Crossover**  
   - Random subtrees are swapped between two parent expressions.

5. **Mutation**  
   - A random function in the expression tree is **replaced** with another function of the same arity.

6. **Elitism**  
   - The top **N best individuals** (elitism) are carried forward **unchanged**.

7. **Termination Condition**  
   - The evolution **stops when the time budget expires** or **an ideal expression is found**.

---

## **📊 Running the Program**
The script supports **command-line arguments** for executing different questions.

### **💡 Question 1: Evaluate an Expression**
**Command:**
```bash
python main.py -question 1 -expr "(add (mul 3 x) (sub x 5))" -n 10 -x "1 2 3 4 5"
```
**Explanation:**
- Evaluates the given expression `(add (mul 3 x) (sub x 5))` with `n=10` and `x=[1,2,3,4,5]`.

---

### **💡 Question 2: Compute Mean Squared Error (MSE)**
**Command:**
```bash
python main.py -question 2 -expr "(add x 2)" -n 10 -m 5 -data "expression_data.txt"
```
**Explanation:**
- Loads data from `expression_data.txt`.
- Evaluates the expression on the dataset.
- Computes **MSE** between predicted and actual values.

---

### **💡 Question 3: Run Genetic Programming**
**Command:**
```bash
python main.py -question 3 -lambda 100 -n 10 -m 5 -data "expression_data.txt" -time_budget 600
```
**Explanation:**
- Runs **GP with 100 population size**.
- Uses a **10-minute time budget**.
- Trains the algorithm on `expression_data.txt`.

---

## **⚙️ Parameters in Genetic Programming**
| Parameter | Description |
|-----------|-------------|
| `Lambda` | Population size |
| `n` | Number of data points |
| `m` | Number of features |
| `data_file` | Path to data file |
| `time_budget` | Maximum time allowed for training |
| `depth` | Maximum depth of expression trees |
| `tree_type_ratio` | Ratio of perfect vs. non-perfect trees |
| `crossover_rate` | Probability of crossover happening |
| `mutation_rate` | Probability of mutation happening |
| `elitism_count` | Number of best individuals retained |

---

## **📈 Performance Optimization**
### **💡 Strategies Used**
1. **Mathematical Safeguards**
   - **Prevent division by zero** (`div` operator).
   - **Handle extreme exponents** (`pow` operator).
   - **Restrict invalid logs & roots** (`log`, `sqrt`).

2. **Tree Pruning**
   - Removes redundant subtrees **after crossover/mutation**.

3. **Parallel Execution**
   - Uses **NumPy vectorized operations** for **faster MSE computation**.

---

## **📌 Summary**
- ✅ This project uses **Genetic Programming (GP)** to evolve mathematical expressions.
- ✅ The algorithm **optimizes symbolic functions** using **selection, crossover, mutation, and elitism**.
- ✅ **Multiple mathematical operators** are implemented for flexibility.
- ✅ The **command-line interface (CLI)** allows flexible execution.
- ✅ The system **ensures numerical stability** with robust error handling.

---

## **🔧 Future Improvements**
- Implement **multi-objective optimization** to balance accuracy and complexity.
- Optimize **execution speed** using GPU-accelerated computation.
- Integrate **Hybrid Genetic Algorithms** combining **GP with Simulated Annealing**.

---

## **📚 References**
- Koza, J. R. (1992). *Genetic Programming: On the Programming of Computers by Means of Natural Selection*.
- Banzhaf, W., Nordin, P., Keller, R. E., & Francone, F. D. (1998). *Genetic Programming: An Introduction*.

---

## **🛠 Requirements**
Install dependencies before running:
```bash
pip install -r requirements.txt
```
---
🔥 **This README provides everything you need to understand and run the project!** 🚀
