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

# Seaborn styles
sns.set()

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
    other_job_order = []
    child = np.zeros(len(parent1))
    for j in parent2:
        if j != select_job:
            other_job_order.append(j)
    k = 0
    for i,j in enumerate(parent1):
        if j == select_job:
            child[i] = j
        else:
            child[i] = other_job_order[k]
            k += 1
    
    return child
    
def mutation(childlist, mutation_rate, num_mutation_jobs, p_t, m_seq):
    for i, chromosome in enumerate(childlist):
        sample_prob = np.random.rand()
        if sample_prob <= mutation_rate:
            mutationpoints = np.random.choice(len(chromosome), num_mutation_jobs, replace = False)
            all_permutations = list(permutations(mutationpoints))
            chrom_copy = copy.deepcopy(chromosome)
            min_makespan = 99999999
            for j in range(1,len(all_permutations)):
                for i in range(len(mutationpoints)):
                    chromosome[mutationpoints[i]] = chrom_copy[all_permutations[j][i]]

                makespan = compute_makespan(chromosome, p_t, m_seq)
                if(makespan < min_makespan):
                    childlist[i] = chromosome
                    min_makespan = makespan
            
        
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
    
    return childlist           

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

def get_critical_path(chromosome,p_t, m_seq):
    critical_path = []
    start_t = np.zeros(len(chromosome))
    end_t = np.zeros(len(chromosome))

    op_count = np.zeros(p_t.shape[0], dtype = np.int32)
    j_time = np.zeros(p_t.shape[0])
    m_time = np.zeros(p_t.shape[1])

    for i,j in enumerate(chromosome):
        completion_t = max(j_time[j], m_time[m_seq[j,op_count[j]]]) + p_t[j,op_count[j]]
        start_t[i] = max(j_time[j], m_time[m_seq[j,op_count[j]]])
        end_t[i] = completion_t
        j_time[j] = completion_t
        m_time[m_seq[j,op_count[j]]] = completion_t
        op_count[j] += 1

    makespan = max(j_time)
    last_end_t = makespan
    for i in range(len(chromosome) - 1, -1, -1):
        if end_t[i] == last_end_t:
            critical_path.insert(0, i)
            last_end_t = start_t[i]

    return critical_path
    
def get_neighbors(chromosome, indices, m_seq):
    # Generate neighboring solutions by swapping pairs of operations
    neighbors = []
    op_count = np.zeros(m_seq.shape[0], np.int32)
    m_list = []
    for j in chromosome:
        m_list.append(m_seq[j,op_count[j]])
        op_count[j] += 1
    
    for i in range(len(indices)-1):
        neighbor = chromosome[:]
        if m_list[indices[i]] == m_list[indices[i+1]]:
            neighbor[indices[i]], neighbor[indices[i+1]] = neighbor[indices[i+1]], neighbor[indices[i]]
        neighbors.append(neighbor)

    return neighbors

def tabu_search(population_list, makespan_list, m_seq, max_iterations, tabu_tenure):
    num_all_ts = int(0.1*len(population_list))
    zipped = list(zip(makespan_list, population_list))
    sorted_zipped = sorted(zipped, key=lambda x: x[0])
    sorted_makespans, sorted_population = zip(*sorted_zipped)
    ts_population = np.asarray(sorted_population)[:num_all_ts]
    non_ts_population = np.asarray(sorted_population)[num_all_ts:]

    tabu_list = []
    for i, chromosome in enumerate(ts_population):
        current_solution = chromosome
        best_makespan = makespan_list[i]
        best_solution = chromosome
        unimprovement = 0
        for iteration in range(max_iterations):
            critical_path = get_critical_path(current_solution, p_t, m_seq)
            neighbors = get_neighbors(current_solution, critical_path, m_seq)
            best_neighbor = None
            best_neighbor_makespan = float('inf')
            
            for neighbor in neighbors:
                if not any(np.array_equal(neighbor, tabu) for tabu in tabu_list):
                    neighbor_makespan = compute_makespan(neighbor, p_t, m_seq)
                    if neighbor_makespan < best_neighbor_makespan:
                        best_neighbor = neighbor
                        best_neighbor_makespan = neighbor_makespan
                        
            
            if best_neighbor is not None:
                current_solution = best_neighbor
                current_makespan = best_neighbor_makespan

                # Update the best solution found
                if current_makespan < best_makespan:
                    best_solution = current_solution
                    best_makespan = current_makespan
                    ts_population[i] = best_solution
                    
                else:
                    unimprovement += 1

                if unimprovement > 20:
                    break
                # Update the tabu list
                tabu_list.append(current_solution)
                if len(tabu_list) > tabu_tenure:
                    tabu_list.pop(0)  # Remove oldest tabu entry
            else:
                break 
    childlist = np.concatenate((ts_population, non_ts_population), axis = 0)
    return  population_list, best_makespan, best_solution

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
    df = pd.read_excel("./Data/lower_bounds.xlsx", sheet_name="orb")
    time_list = []
    ratio_list = []
    outDf = pd.DataFrame()
    for index in range(1,11):
        instance_name = "orb0"+str(index)
        if(index >= 10):instance_name = "orb"+str(index)
        j, m, p_t, m_seq = read_file(instance_name)
        population_size = 200
        population_list = generate_init_pop(population_size, j ,m)
        crossover_rate = 0.95
        mutation_rate = 0.15
        max_mutatio_rate = 0.4
        num_iteration = 1000
        min_makespan_record = []
        avg_makespan_record = []
        min_makespan = 9999999
        best_maksapn_not_changed = 0
        begint = time.time()

        for i in tqdm(range(num_iteration)):
            parentlist, childlist = job_order_crossover(population_list, j, crossover_rate)
            childlist = mutation(childlist, mutation_rate, 3, p_t, m_seq)
            population_list = np.concatenate((parentlist, childlist), axis=0)
            makespan_list = np.zeros(len(population_list))
            for k in range(len(population_list)):
                makespan_list[k] = compute_makespan(population_list[k], p_t, m_seq)
                if makespan_list[k] < min_makespan:
                    min_m = makespan_list[k]
                    min_makespan = makespan_list[k]
                    min_c = population_list[k]
                    best_maksapn_not_changed = 0
                else:
                    best_maksapn_not_changed += 1
   
            population_list = binary_selection(population_list, makespan_list)
            population_list, min_ts_makespan, min_ts_chromosome = tabu_search(population_list, makespan_list, m_seq, 1000, 9)
            if min_ts_makespan < min_makespan:
                min_makespan = min_ts_makespan
                min_c = min_ts_chromosome

            #mutation_rate = min(mutation_rate * (1+0.1*best_maksapn_not_changed/num_iteration),max_mutatio_rate)
            min_makespan_record.append(min_makespan)
            avg_makespan_record.append(np.average(makespan_list))

        if index == 1:    
            plt.plot(avg_makespan_record)
            plt.plot(min_makespan_record)
            plt.savefig("orb"+str(index))
        time_consume = time.time() - begint
        ratio = (min_makespan - df["lower bound"][index-1])/df["lower bound"][index-1]
        time_list.append(time)
        ratio_list.append(ratio)
        print(min_makespan, " ", df["lower bound"][index-1])
        draw_gantt_chart(min_c, p_t, m_seq, "Gantt Chart for GA on orb" + str(index)+ " instance")
        row_input = pd.DataFrame([[min_makespan, df["lower bound"][index-1], time_consume]], columns = ["GA","Optimal","time(sec)"])
        outDf = pd.concat([outDf, row_input], ignore_index=True)

    outDf.to_csv("Result.csv")
    draw_bar_plot("Result.csv", ["orb"+str(i+1) for i in range(1, 11)])

    
    

        



        





    
