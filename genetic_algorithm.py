import pandas as pd
import numpy as np
import copy
import math
import time
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import permutations
import random


def read_file(filename):
    path = 'Data/'+filename
    f = open(path).read().splitlines()

    for i,line in enumerate(f):
        line = line.split()
        if i == 4:
            j = int(line[0])
            m = int(line[1])
            p_t = np.zeros((j,m))
            m_seq = np.zeros((j,m), dtype = np.int32)
        elif i > 4:
            for k in range(len(line)):
                if k % 2 == 0:
                    m_seq[i-5,int(k/2)] = int(line[k])
                elif k % 2 == 1:
                    p_t[i-5,int(k/2)] = int(line[k])
    
    return int(j), int(m), p_t, m_seq

def compute_makespan(chromosome, p_t, m_seq):
    op_count = np.zeros(p_t.shape[0], dtype = np.int32)
    j_time = np.zeros(p_t.shape[0])
    m_time = np.zeros(p_t.shape[1])
    for j in chromosome:
        completion_t = max(j_time[j], m_time[m_seq[j,op_count[j]]]) + p_t[j,op_count[j]]
        j_time[j] = completion_t
        m_time[m_seq[j,op_count[j]]] = completion_t
        op_count[j] += 1

    makespan = max(j_time)
    
    return makespan
            
def generate_init_pop(population_size, j, m):
    population_list = np.zeros((population_size, int(j*m)), dtype = np.int32)
    chromosome = np.zeros(j*m)
    start = 0
    for i in range(j):
        chromosome[start:start+m] = i
        start += m

    for i in range(population_size):
        np.random.shuffle(chromosome)
        population_list[i] = chromosome
    
    return population_list

def job_order_crossover(populationlist, j, crossover_rate):
    parentlist = copy.deepcopy(populationlist)
    childlist = copy.deepcopy(populationlist)
    for i in range(len(parentlist),2):
        sample_prob=np.random.rand()
        if sample_prob <= crossover_rate:
            parent_id = np.random.choice(len(populationlist), 2, replace=False)
            select_job = np.random.choice(j, 1, replace=False)[0]
            child_1 = job_order_implementation(parentlist[parent_id[0]], parentlist[parent_id[1]], select_job)
            child_2 = job_order_implementation(parentlist[parent_id[1]], parentlist[parent_id[0]], select_job)
            childlist[i] = child_1
            childlist[i+1] = child_2

    return parentlist, childlist

def job_order_implementation(parent1, parent2, select_job):
    child = np.empty(len(parent1), dtype=int)
    
    # 提取 parent2 中所有非 select_job 的作業順序
    other_jobs = [j for j in parent2 if j != select_job]
    
    k = 0
    for i, j in enumerate(parent1):
        if j == select_job:
            child[i] = j
        else:
            child[i] = other_jobs[k]
            k += 1

    return child
    
def mutation(childlist, mutation_rate, num_mutation_jobs, p_t, m_seq):
    for idx, chromosome in enumerate(childlist):
        random_values = np.random.rand(2)
        if random_values[0] <= mutation_rate:
            mutated = chromosome.copy()
            # permutation-based mutation
            if random_values[1] <= 0.1:
                mutationpoints = np.random.choice(len(chromosome), num_mutation_jobs, replace=False)
                all_permutations = list(permutations(mutationpoints))

                best_chromosome = mutated
                min_makespan = compute_makespan(mutated, p_t, m_seq)

                for perm in all_permutations[1:]:
                    new_chrom = mutated.copy()
                    for i in range(len(mutationpoints)):
                        new_chrom[mutationpoints[i]] = mutated[perm[i]]
                    makespan = compute_makespan(new_chrom, p_t, m_seq)
                    if makespan < min_makespan:
                        min_makespan = makespan
                        best_chromosome = new_chrom
                        mutated = best_chromosome
            else:
                mutationpoints = np.random.choice(len(chromosome), int(p_t.shape[0]*p_t.shape[1]*0.05), replace=False)
                origin = mutationpoints.copy()
                new_chrom = mutated.copy()
                np.random.shuffle(mutationpoints)
                for i in range(len(mutationpoints)):
                    mutated[mutationpoints[i]] = new_chrom[origin[i]]

            childlist[idx] = mutated

    # Compute makespans
    makespan_list = np.array([compute_makespan(chrom, p_t, m_seq) for chrom in childlist])

    # Select top 90%
    num_all_mut = min(int(0.1 * len(childlist)), len(childlist) - 1)
    sorted_indices = np.argsort(makespan_list)
    partial_mut_id = sorted_indices[:-num_all_mut]

    # Inject 10% new individuals
    all_mut = generate_init_pop(num_all_mut, p_t.shape[0], p_t.shape[1])
    retained = [copy.deepcopy(childlist[i]) for i in partial_mut_id]

    # Combine and return new population
    return np.array(retained + list(all_mut))

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

def get_critical_blocks(chromosome, p_t, m_seq):
    op_count = np.zeros(p_t.shape[0], dtype=np.int32)
    j_time = np.zeros(p_t.shape[0])
    m_time = np.zeros(p_t.shape[1])
    start_t = np.zeros(len(chromosome))
    end_t = np.zeros(len(chromosome))
    m_list = []

    for i, job in enumerate(chromosome):
        m_id = m_seq[job, op_count[job]]
        m_list.append(m_id)
        start = max(j_time[job], m_time[m_id])
        end = start + p_t[job, op_count[job]]
        start_t[i] = start
        end_t[i] = end
        j_time[job] = end
        m_time[m_id] = end
        op_count[job] += 1

    makespan = max(j_time)
    critical_path = []
    last_end = makespan
    for i in range(len(chromosome)-1, -1, -1):
        if end_t[i] == last_end:
            critical_path.insert(0, i)
            last_end = start_t[i]

    # 找出 block: 關鍵路徑中，同一機台連續 operation
    blocks = []
    if not critical_path:
        return [], makespan, m_list

    start_idx = 0
    for i in range(1, len(critical_path)):
        curr_idx = critical_path[i]
        prev_idx = critical_path[i-1]
        if m_list[curr_idx] != m_list[prev_idx]:
            if i - start_idx >= 2:
                blocks.append(critical_path[start_idx:i])
            start_idx = i
    if len(critical_path) - start_idx >= 2:
        blocks.append(critical_path[start_idx:])

    return blocks, makespan, m_list

def apply_move(chrom, i, j):
    new_chrom = chrom.copy()
    new_chrom[i], new_chrom[j] = new_chrom[j], new_chrom[i]
    return new_chrom

def tabu_search_ns1996(chromosome, p_t, m_seq, max_iter=100, tabu_tenure=7):
    current = chromosome.copy()
    best = current.copy()
    best_makespan = compute_makespan(best, p_t, m_seq)
    tabu_list = []

    for _ in range(max_iter):
        blocks, _, m_list = get_critical_blocks(current, p_t, m_seq)
        found_better = False
        move_applied = None

        for block in blocks:
            if len(block) < 2:
                continue

            candidate_moves = []
            # Swap first two
            i1, i2 = block[0], block[1]
            if (i1, i2) not in tabu_list:
                candidate_moves.append((i1, i2))

            # Swap last two
            j1, j2 = block[-2], block[-1]
            if (j1, j2) not in tabu_list and (j1, j2) != (i1, i2):
                candidate_moves.append((j1, j2))

            for i, j in candidate_moves:
                neighbor = apply_move(current, i, j)
                new_makespan = compute_makespan(neighbor, p_t, m_seq)
                if new_makespan < best_makespan:
                    best = neighbor
                    best_makespan = new_makespan
                    current = neighbor
                    move_applied = (i, j)
                    found_better = True
                    break
            if found_better:
                break

        if found_better and move_applied:
            tabu_list.append(move_applied)
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)
        else:
            break  # No improving move found → local optimum

    return best, best_makespan

def compute_diversity(population):
    unique = np.unique(population, axis=0)
    return len(unique) / len(population)

def intensify_top10_with_tabu(population_list, p_t, m_seq, tabu_iter=100, tabu_tenure=7):
    """
    對前 10% makespan 最佳的染色體進行 Tabu Search 強化，
    並回傳：
        - 強化後的 population
        - 最佳 makespan
        - 對應的基因序列（chromosome）
    """
    population_size = len(population_list)
    makespan_list = np.array([compute_makespan(chrom, p_t, m_seq) for chrom in population_list])
    
    # 取前 10% 的 index
    num_to_improve = max(1, int(0.1 * population_size))
    best_indices = np.argsort(makespan_list)[:num_to_improve]

    for idx in best_indices:
        improved, improved_makespan = tabu_search_ns1996(population_list[idx], p_t, m_seq, max_iter=tabu_iter, tabu_tenure=tabu_tenure)
        population_list[idx] = improved
        makespan_list[idx] = improved_makespan

    # 找到最佳解
    min_idx = np.argmin(makespan_list)
    min_makespan = makespan_list[min_idx]
    best_chromosome = population_list[min_idx]

    return population_list, min_makespan, best_chromosome

def draw_bar_plot(filename, listx):
    df = pd.read_csv(filename)
    # Data for the bar plot

    # Set bar width and adjust positions for grouped bars
    width = 0.25
    listx1 = [x - (width / 2) for x in range(len(listx))]  # Shift left
    listx2 = [x + (width / 2) for x in range(len(listx))]  # Shift right

    # Y-axis data
    listy1 = df["GA"]
    listy2 = df["Optimal"]

    plt.bar(listx1, listy1, width, label="GA")
    plt.bar(listx2, listy2, width, label="Opt")

    # Set x-ticks and labels
    plt.xticks(range(len(listx)), labels=listx)

    # Add legend, title, and axis labels
    plt.legend()
    plt.title("Makespan Comparison on OR-Library Instances")
    plt.ylabel("Makesapn")
    plt.xlabel("15*15 OR Library Instances")

    # Show the plot
    plt.savefig("Makespan Comparison_on OR-Library Instances Bar Chart")
    plt.show()

def draw_gantt_chart(chromosome, p_t, m_seq, title):
    gantt_data = []
    start_t = np.zeros(p_t.shape[0])  # Tracks the start time for each job
    m_time = np.zeros(p_t.shape[1])   # Tracks the time at which each machine will be free
    j_time = np.zeros(p_t.shape[0])   # Tracks the time at which each job will finish
    op_count = np.zeros(p_t.shape[0], dtype=np.int32)  # Operation counter for each job

    # Iterate through the chromosome to generate Gantt chart data
    for i, gene in enumerate(chromosome):
        job = gene  # The current job from the chromosome
        machine = m_seq[job, op_count[job]]  # Get the machine for the current operation

        # Calculate the start and completion time of the current operation
        start_time = max(j_time[job], m_time[machine])
        completion_time = start_time + p_t[job, op_count[job]]

        # Update job and machine available times
        j_time[job] = completion_time
        m_time[machine] = completion_time

        # Append the Gantt chart data: job, start, finish, machine
        gantt_data.append((f'Job {job}', start_time, completion_time, f'Machine {machine+1}'))
        op_count[job] += 1

    # Convert Gantt chart data into a DataFrame for easier plotting
    df = pd.DataFrame(gantt_data, columns=['Job', 'Start', 'Finish', 'Machine'])

    df['Machine_ID'] = df['Machine'].apply(lambda x: int(x.split(' ')[1]))  # Extract numeric ID
    df = df.sort_values(by='Machine_ID')  # Sort by machine ID

    # Plot the Gantt chart
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create Gantt chart bars grouped by machine
    for i, row in df.iterrows():
        ax.barh(row['Machine'], row['Finish'] - row['Start'], left=row['Start'], height=0.4, label=row['Job'])

    # Add labels and formatting
    ax.set_xlabel('Time')
    ax.set_ylabel('Machine')
    ax.set_title(title)

    plt.savefig("GanttPlot/"+title)

if __name__ == "__main__":
    problem = "la"
    start = 26
    end = 31
    df = pd.read_excel("./Data/lower_bounds.xlsx", sheet_name = problem)
    time_list = []
    ratio_list = []
    outDf = pd.DataFrame()
    for index in range(start,end):
        instance_name = problem+"0"+str(index)
        if(index >= 10):instance_name = problem+str(index)
        j, m, p_t, m_seq = read_file(instance_name)
        population_size = 100
        population_list = generate_init_pop(population_size, j ,m)
        crossover_rate = 0.95
        mutation_base_rate = 0.15
        scale = 0.3
        max_mutatio_rate = 0.4
        num_iteration = 200
        min_makespan_record = []
        avg_makespan_record = []
        min_makespan = 9999999
        begint = time.time()

        for i in tqdm(range(num_iteration)):
            parentlist, childlist = job_order_crossover(population_list, j, crossover_rate)
            diversity = compute_diversity(population_list)
            mutation_rate = mutation_base_rate + (1 - diversity) * scale
            childlist = mutation(childlist, mutation_rate, int(j*m*0.025) , p_t, m_seq)
            population_list = np.concatenate((parentlist, childlist), axis=0)
            makespan_list = np.zeros(len(population_list))
            for k in range(len(population_list)):
                makespan_list[k] = compute_makespan(population_list[k], p_t, m_seq)
                if makespan_list[k] < min_makespan:
                    min_m = makespan_list[k]
                    min_makespan = makespan_list[k]
                    min_c = population_list[k]

            population_list = binary_selection(population_list, makespan_list)
            population_list, min_ts_makespan, min_ts_chromosome = intensify_top10_with_tabu(population_list, p_t, m_seq)
            if min_ts_makespan < min_makespan:
                min_makespan = min_ts_makespan
                min_c = min_ts_chromosome

            min_makespan_record.append(min_makespan)
            avg_makespan_record.append(np.average(makespan_list))

        if index == start:    
            plt.plot(avg_makespan_record)
            plt.plot(min_makespan_record)
            plt.savefig("orb"+str(index))
        time_consume = time.time() - begint
        ratio = (min_makespan - df["lower bound"][index-1])/df["lower bound"][index - 1]
        time_list.append(time_consume)
        ratio_list.append(ratio)
        print(min_makespan, " ", df["lower bound"][index- 1])
        draw_gantt_chart(min_c, p_t, m_seq, "Gantt Chart for GA on orb" + str(index)+ " instance")
        row_input = pd.DataFrame([[min_makespan, df["lower bound"][index-1], time_consume]], columns = ["GA","Optimal","time(sec)"])
        outDf = pd.concat([outDf, row_input], ignore_index=True)

    outDf.to_csv("Result.csv", index = False)
    draw_bar_plot("Result.csv", ["la"+str(i+1) for i in range(start, end)])

    
    

        



        





    
