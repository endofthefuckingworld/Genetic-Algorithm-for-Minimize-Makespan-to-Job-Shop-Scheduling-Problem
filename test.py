from collections import defaultdict

def permissible_left_shift(job_sequence, job_operations):
    """
    Generate an active schedule and corresponding active chromosome.
    
    Parameters:
    - job_sequence: List of integers representing the job sequence (chromosome).
    - job_operations: Dictionary where keys are job IDs and values are lists of tuples 
                      (machine_id, processing_time) for each operation of the job.

    Returns:
    - active_chromosome: List of integers representing the updated job sequence (active chromosome).
    - active_schedule: List of tuples (job, operation_index, start_time, end_time, machine_id).
    """
    # Initialize machine availability and job progress
    machine_availability = defaultdict(int)  # Machine -> Next available time
    job_progress = defaultdict(int)         # Job -> Next operation index
    active_schedule = []                    # List of scheduled operations
    active_chromosome = []                  # Reordered job sequence

    for job in job_sequence:
        # Get the current operation of the job
        op_index = job_progress[job]
        machine_id, proc_time = job_operations[job][op_index]
        
        # Determine start time: max of machine availability and job's precedence constraints
        start_time = max(machine_availability[machine_id], 
                         active_schedule[-1][3] if active_schedule and active_schedule[-1][0] == job else 0)
        end_time = start_time + proc_time
        
        # Schedule the operation
        active_schedule.append((job, op_index + 1, start_time, end_time, machine_id))
        active_chromosome.append(job)  # Add the job to the reordered chromosome
        
        # Update machine availability and job progress
        machine_availability[machine_id] = end_time
        job_progress[job] += 1
    
    return active_chromosome, active_schedule

# Example Input
job_operations = {
    1: [(1, 3), (2, 1), (3, 2)],  # Job 1: (Machine 1 -> 3), (Machine 2 -> 1), (Machine 3 -> 2)
    2: [(3, 1), (1, 5), (2, 3)],  # Job 2: (Machine 3 -> 1), (Machine 1 -> 5), (Machine 2 -> 3)
    3: [(2, 3), (3, 2), (1, 3)]   # Job 3: (Machine 2 -> 3), (Machine 3 -> 2), (Machine 1 -> 3)
}

job_sequence = [2, 3, 1, 1, 2, 2, 3, 1, 3]

# Run the scheduling algorithm
active_chromosome, active_schedule = permissible_left_shift(job_sequence, job_operations)

# Display the active chromosome
print("Active Chromosome:")
print(active_chromosome)

# Display the active schedule
print("\nActive Schedule:")
print("(Job, Operation, Start Time, End Time, Machine)")
for operation in active_schedule:
    print(operation)
