# GA-Job-Shop-Scheduling
Simple Implementation of Applying Genetic Algorithm to JSSP in Python.


## :black_nib: Genetic Algorithm Background
Genetic algorithms are a type of optimization algorithm inspired by the process of natural selection. They are used to find approximate solutions to complex optimization and search problems.The procedure of applying a genetic algorithm to solve a Job Shop Scheduling Problem (JSSP) involves several steps. Here's an overview of the typical procedure:


### :arrow_down_small: JSSP Solution Encoding <br>
In the context of the Job Shop Scheduling Problem (JSSP), encoding refers to how you represent a solution to the problem using a data structure that a computer can work with. One common method is encoding a solution as a sequence of jobs. In this encoding scheme, a chromosome is comprised of a sequence of integers, with each integer representing an operation of a job. The order of these integers within the chromosome specifies the sequence in which operations are to be executed on the machines.

**Example:**
Let's consider the chromosome [2, 3, 2, 1, 1, 3, 2, 3, 1], where 1, 2, and 3 correspond to different jobs. In this encoding, The first gene (2) represents the first operation of Job 2,the second gene (3) represents the first operation of Job 3 and the third gene (2) represents the second operation of Job 2.

### :arrow_down_small: Initial Population Generation <br>
Generate an initial population of schedules. Each schedule represents a possible solution to the JSSP. You can initialize the population randomly or using domain-specific knowledge such as dispatching rules or other heuristic methods. We use randomly initialize the population here.
```python
def generate_init_pop(population_size, j, m):
    population_list = np.zeros((population_size, int(j*m)), dtype = np.int32)
    chromosome = np.zeros(j*m)
    start = 0
    for i in range(j): #every job needs m operations
        chromosome[start:start+m] = i
        start += m

    for i in range(population_size): #Disrupting the order of job sequences.
        np.random.shuffle(chromosome)
        population_list[i] = chromosome
    
    return population_list
```
