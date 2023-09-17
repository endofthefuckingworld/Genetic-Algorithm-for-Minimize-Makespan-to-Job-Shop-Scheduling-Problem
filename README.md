# GA-Job-Shop-Scheduling
Simple Implementation of Applying Genetic Algorithm to JSSP in Python.


## :black_nib: Genetic Algorithm Background
Genetic algorithms are a type of optimization algorithm inspired by the process of natural selection. They are used to find approximate solutions to complex optimization and search problems.The procedure of applying a genetic algorithm to solve a Job Shop Scheduling Problem (JSSP) involves several steps. Here's an overview of the typical procedure:

  

### :arrow_down_small: JSSP Solution Encoding <br>
In the context of the Job Shop Scheduling Problem (JSSP), encoding refers to how you represent a solution to the problem using a data structure that a computer can work with. One common method is encoding a solution as a sequence of jobs ([M. Gen, et al.(1994)](https://ieeexplore.ieee.org/document/400072/)). In this encoding scheme, a chromosome is comprised of a sequence of integers, with each integer representing an operation of a job. The order of these integers within the chromosome specifies the sequence in which operations are to be executed on the machines.

**Example:**
Let's consider the chromosome [2, 3, 2, 1, 1, 3, 2, 3, 1], where 1, 2, and 3 correspond to different jobs. In this encoding, The first gene (2) represents the first operation of Job 2,the second gene (3) represents the first operation of Job 3 and the third gene (2) represents the second operation of Job 2.
<br>
<div align=center>
<img src="https://github.com/endofthefuckingworld/Genetic-Algorithm-for-Minimize-Makespan-to-Job-Shop-Scheduling-Problem/blob/main/Picture/encoding.gif" width="780" height="420">
</div>
<br>

### :arrow_down_small: Fitness Evaluation <br>
Evaluate the fitness(Minimized Makespan) of each schedule in the population.According to the formulation of JSSP, The primary constraints in the JSSP include precedence constraints and machine sharing constraints:  
1. Precedence Constraints: certain operations must be performed in a specific order. This means that certain jobs cannot start until others are completed.
2. Machine Sharing Constraints: Each machine has its own availability and can only process one operation at a time. This constraint ensures that no machine is overutilized or double-booked.  
Therefore,the completion time of an operation must be greater to the process time plus completion time of the machine that operation assigned to and the previous operation of the same job. We can use this way to evaluate the makespan of the chromosome.


```python
def compute_makespan(chromosome, p_t, m_seq):
    op_count = np.zeros(p_t.shape[0], dtype = np.int32)
    j_time = np.zeros(p_t.shape[0])
    m_time = np.zeros(p_t.shape[1])

    for j in chromosome:
        completion_t = max(j_time[j], m_time[m_seq[j,op_count[j]]]) + p_t[j,op_count[j]] #Precedence constraints and machine sharing constraints
        j_time[j] = completion_t
        m_time[m_seq[j,op_count[j]]] = completion_t
        op_count[j] += 1

    makespan = max(j_time)

    return makespan
```
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
### :arrow_down_small: Crossover <br>
Apply crossover operators to pairs of parent schedules to create new child schedules. In JSSP, specialized crossover operators like Job-Order Crossover (JOX) or Precedence Preservative Crossover (PPX) are often used to maintain the problem's constraints. We use Job-Order Crossover here according to ([Lamos-Díaz, Henry, et al.(2017)](http://www.scielo.org.co/scielo.php?pid=S0121-11292017000100113&script=sci_arttext)).

**Job Order Crossover**
1. A pair of parents (chromosomes) are randomly selected from the population pool.
2. Ramdomly select one job j.  
3. Copy all job j from parent 1 to child 1 with the same position.  
4. The remaining empty positions in child 1 are filled with the genes of parent 2 that are different from the job j.

<br>
<div align=center>
<img src="https://github.com/endofthefuckingworld/Genetic-Algorithm-for-Minimize-Makespan-to-Job-Shop-Scheduling-Problem/blob/main/Picture/crossover.gif" width="780" height="420">
</div>
<br>


```python
def job_order_crossover(populationlist, j, crossover_rate):
    parentlist = copy.deepcopy(populationlist)
    childlist = copy.deepcopy(populationlist)
    for i in range(len(parentlist),2):
        sample_prob=np.random.rand()
        if sample_prob <= crossover_rate:
            parent_id = np.random.choice(len(populationlist), 2, replace=False) #A pair of parents (chromosomes) are randomly selected from the population pool.
            select_job = np.random.choice(j, 1, replace=False)[0] #Ramdomly select one job j.
            child_1 = job_order_implementation(parentlist[parent_id[0]], parentlist[parent_id[1]], select_job)
            child_2 = job_order_implementation(parentlist[parent_id[1]], parentlist[parent_id[0]], select_job)
            childlist[i] = child_1
            childlist[i+1] = child_2

    return parentlist, childlist

def job_order_implementation(parent1, parent2, select_job):
    other_job_order = []
    child = np.zeros(len(parent1))
    for j in parent2:
        if j != select_job:
            other_job_order.append(j)
    k = 0
    for i,j in enumerate(parent1):
        if j == select_job:
            child[i] = j  #Copy all job j from parent1 to child1 with the same position.
        else:
            child[i] = other_job_order[k]  #The remaining empty positions are filled with the genes of parent2 that are different from the job j.
            k += 1
    
    return child
```

### :arrow_down_small: Mutation <br>

Ｍutation introduces additional variability into the population, which helps prevent it from prematurely converging towards a local optimum. We mutate genes through gene shifting and the process is as follows:
  
1. If the sample probability of a child is less or equal than the mutation probability parameter, the mutation is then executed.  
2. Randomly select the genes to be shifted and the number of genes to mutate is based on the mutation selection rate. For example, Each 
chromosome has 36 genes and if the mutation selection rate equals to 0.5, the number of genes to shift is 18.  
3. Perform gene shifting, as illustrated in the diagram.
4. For the last 10% child, replace them with randomly initialized population.

<br>
<div align=center>
<img src="https://github.com/endofthefuckingworld/Genetic-Algorithm-for-Minimize-Makespan-to-Job-Shop-Scheduling-Problem/blob/main/Picture/mutation.gif" width="780" height="320">
</div>
<br>

```python
def mutation(childlist, num_mutation_jobs, mutation_rate, p_t, m_seq):
    for chromosome in childlist:
        sample_prob = np.random.rand()
        if sample_prob <= mutation_rate:
            mutationpoints = np.random.choice(len(chromosome), num_mutation_jobs, replace = False)
            chrom_copy = copy.deepcopy(chromosome)
            for i in range(len(mutationpoints)-1):
                chromosome[mutationpoints[i+1]] = chrom_copy[mutationpoints[i]]

            chromosome[mutationpoints[0]] = chrom_copy[mutationpoints[-1]]
    
    makespan_list = np.zeros(len(childlist))
    for i,chromosome in enumerate(childlist):
        makespan_list[i] = compute_makespan(chromosome, p_t, m_seq)
    
    num_all_mut = int(0.1*len(childlist))
    zipped = list(zip(makespan_list, np.arange(len(makespan_list))))
    sorted_zipped = sorted(zipped, key=lambda x: x[0])
    zipped = zip(*sorted_zipped)
    partial_mut_id = np.asarray(list(zipped)[1])[:-num_all_mut]
    all_mut = generate_init_pop(num_all_mut, p_t.shape[0], p_t.shape[1])
    childlist = np.concatenate((all_mut,copy.deepcopy(childlist)[partial_mut_id]), axis = 0)
```

### :arrow_down_small: Selection <br>
Select schedules from the population to act as parents for the next generation. Common selection methods include roulette wheel selection, tournament selection, and rank-based selection. According to ([Pezzella, Ferdinando, Gianluca Morganti, and Giampiero Ciaschetti (2008)](https://www.sciencedirect.com/science/article/pii/S0305054807000524)), binary tournament gives great results so we decide to use it and the process is as follows:  
1. The top 10% of chromosomes are guaranteed to be kept for the next generation.
2. Two chromosomes are randomly chosen from the population and the best of them is selected for next generation. Keep doing that until the next generation is filled.
   

```python
def binary_selection(populationlist, makespan_list):
    new_population = np.zeros((int(len(populationlist)/2), populationlist.shape[1]), dtype = np.int32)
    
    num_self_select = int(0.1*len(populationlist)/2)
    num_binary = int(len(populationlist)/2) - num_self_select
    zipped = list(zip(makespan_list, np.arange(len(makespan_list))))
    sorted_zipped = sorted(zipped, key=lambda x: x[0])
    zipped = zip(*sorted_zipped)
    self_select_id = np.asarray(list(zipped)[1])[:num_self_select]
    
    for i in range(num_binary):
        select_id = np.random.choice(len(makespan_list), 2, replace=False)
        if makespan_list[select_id[0]] < makespan_list[select_id[1]]:
            new_population[i] = populationlist[select_id[0]]
        else:
            new_population[i] = populationlist[select_id[1]]
    
    new_population[-num_self_select:] = copy.deepcopy(populationlist)[self_select_id]
    
    return new_population
```

### :arrow_down_small: Main function <br>
The following will outline the various steps of implementing a genetic algorithm. First, declare the parameters of the genetic algorithm, then read the file, generate the initial population, perform crossover and mutation, proceed with the selection of the next generation, and repeat this process until all loops are completed. Finally, output the scheduling results and a convergence chart showing the decrease in makespan.
```python
if __name__ == "__main__":
    instance_name = "la01"
    j, m, p_t, m_seq = read_file(instance_name)
    population_size = 100
    population_list = generate_init_pop(population_size, j ,m)
    crossover_rate = 1.0
    mutation_rate = 0.15
    mutation_selection_rate = 0.15
    num_mutation_jobs=round(j*m*mutation_selection_rate)
    num_iteration = 1000
    min_makespan_record = []
    avg_makespan_record = []
    min_makespan = 9999999

    for i in tqdm(range(num_iteration)):
        parentlist, childlist = job_order_crossover(population_list, j, crossover_rate)
        mutation(childlist, num_mutation_jobs, mutation_rate, p_t, m_seq)
        population_list = np.concatenate((parentlist, childlist), axis=0)
        makespan_list = np.zeros(len(population_list))
        for k in range(len(population_list)):
            makespan_list[k] = compute_makespan(population_list[k], p_t, m_seq)
            if makespan_list[k] < min_makespan:
                min_makespan = makespan_list[k]
                best_job_order = population_list[k]
        
        population_list = binary_selection(population_list, makespan_list)
        min_makespan_record.append(min_makespan)
        avg_makespan_record.append(np.average(makespan_list))

    print(min_makespan)
    print(best_job_order)

    import matplotlib.pyplot as plt
    plt.plot([i for i in range(len(min_makespan_record))],min_makespan_record,'b',label='Best')
    plt.plot([i for i in range(len(avg_makespan_record))],avg_makespan_record,'g',label='Average')
    plt.ylabel('makespan',fontsize=15)
    plt.xlabel('generation',fontsize=15)
    plt.legend()
    plt.title("Decreasing of the makespan in "+instance_name)
    plt.show()
```
