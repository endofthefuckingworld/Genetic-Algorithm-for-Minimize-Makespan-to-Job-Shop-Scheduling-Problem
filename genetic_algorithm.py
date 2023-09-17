import pandas as pd
import numpy as np
import copy
import math
from tqdm import tqdm

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

def two_point_crossover(populationlist, crossover_rate):
    parentlist = copy.deepcopy(populationlist)
    childlist = copy.deepcopy(populationlist)
    for i in range(len(parentlist),2):
        sample_prob=np.random.rand()
        if sample_prob <= crossover_rate:
            cutpoint = np.random.choice(2, parentlist.shape[1], replace = False)
            cutpoint.sort()
            parent_1 = parentlist[i]
            parent_2 = parentlist[i+1]
            child_1 = copy.deepcopy(parent_1)
            child_2 = copy.deepcopy(parent_2)
            child_1[cutpoint[0]:cutpoint[1]] = parent_2[cutpoint[0]:cutpoint[1]]
            child_2[cutpoint[0]:cutpoint[1]] = parent_1[cutpoint[0]:cutpoint[1]]
            childlist[i] = child_1
            childlist[i+1] = child_2
    
    return parentlist, childlist

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

def repair(chromosome, j, m):
    job_count = np.zeros(j)
    for j in chromosome:
        job_count[j] += 1
    
    job_count = job_count - m

    much_less = [[],[]]
    is_legall = True
    for j,count in enumerate(job_count):
        if count > 0:
            is_legall = False
            much_less[0].append(j)
        elif count < 0:
            is_legall = False
            much_less[1].append(j)

    if is_legall == False:
        for m in much_less[0]:
            for j in range(len(chromosome)):
                if chromosome[j] == m:
                    less_id = np.random.choice(len(much_less[1]),1)[0]
                    chromosome[j] = much_less[1][less_id]
                    job_count[m] -= 1
                    job_count[much_less[1][less_id]] += 1

                    if job_count[much_less[1][less_id]] == 0:
                        much_less[1].remove(much_less[1][less_id])

                    if job_count[m] == 0:
                        break
    
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
    
def selection(populationlist, makespan_list):
    num_self_select = int(0.2*len(populationlist)/2)
    num_roulette_wheel = int(len(populationlist)/2) - num_self_select
    zipped = list(zip(makespan_list, np.arange(len(makespan_list))))
    sorted_zipped = sorted(zipped, key=lambda x: x[0])
    zipped = zip(*sorted_zipped)
    self_select_id = np.asarray(list(zipped)[1])[:num_self_select]
    
    makespan_list = 1/makespan_list
    selection_prob = makespan_list/sum(makespan_list)
    roulette_wheel_id = np.random.choice(len(populationlist), size = num_roulette_wheel, p = selection_prob)
    new_population = np.concatenate((copy.deepcopy(populationlist)[self_select_id],copy.deepcopy(populationlist)[roulette_wheel_id]), axis=0)
    
    return new_population

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
    

        



        





    
