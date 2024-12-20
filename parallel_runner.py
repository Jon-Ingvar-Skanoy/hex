# parallel_runner.py

import os
import subprocess
import time
import itertools

if __name__ == "__main__":
    
    # Parameter grids
    board_sizes = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    sample_sizes = [1000, 10000]
    number_of_clauses_list = [1000, 10000, 100000]
    s_values = [1.0] # 0.5, , 2.0
    T_ratios = [0.8]  # 1.2
    q_values = [1.0] #0.5, 
    depth_values = [1] #, 2, 3
    delay = 2

    max_concurrent_processes = 12

    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)

    
    exp_name = "initial"
    # sample_size = 10000
    open_position = 40
    moves_before = 0
    message_size = 128
    message_bits_ratio = 2
    double_hashing = False


    processes = []
    parameter_combinations = list(itertools.product(
        board_sizes, sample_sizes, number_of_clauses_list,
        s_values, T_ratios, q_values, depth_values
    ))

    
    for params in parameter_combinations:
        (board_size, sample_size, number_of_clauses,s, T_ratio, q, depth) = params

        T = int(number_of_clauses * T_ratio)  # Calculate T based on ratio
        param_str = f"b{board_size}_samp{sample_size}_c{number_of_clauses}_T{T}_s{s}_q{q}_d{depth}"
        dh_string = "dh2" if double_hashing else "dh1"

        log_file_path = f"logs/{exp_name}_{param_str}_{dh_string}.log"
        result_file_path = f"results/experiments/{exp_name}_{param_str}_{dh_string}.txt"

        if os.path.exists(log_file_path) or os.path.exists(result_file_path):
            print(f"Skipping already processed combination: {param_str}")
            continue
        
        command = [
            "python", "experiment_initial.py",
            "--board_size", str(board_size),
            "--experiment_name", str(exp_name),
            "--sample_size", str(sample_size),
            "--open_position", str(open_position),
            "--moves_before", str(moves_before),
            "--message_size", str(message_size),
            "--message_bits", str(message_bits_ratio),
            "--double_hashing", str(double_hashing),
            "--epochs", "150",
            "--number_of_clauses", str(number_of_clauses),
            "--T", str(T),
            "--s", str(s),
            "--q", str(q),
            "--depth", str(depth)
        ]

        print(f"Starting process for {param_str} in {delay} seconds..")
        process = subprocess.Popen(
            command,
            stdout=open(log_file_path, "w"),
            stderr=subprocess.STDOUT,
            text=True
        )
        time.sleep(delay)
        processes.append((board_size, process))
        
        # Check running processes and limit concurrency
        while len(processes) >= max_concurrent_processes:
            time.sleep(delay)  # Wait for a second before checking again
            # Remove completed processes
            for param_str, proc in processes[:]:
                if proc.poll() is not None:  # Process has finished
                    processes.remove((param_str, proc))
                    print(f"Process for {param_str} completed.")

    # Wait for all remaining processes to complete
    for param_str, proc in processes:
        proc.wait()
        print(f"Process for {param_str} completed.")

    print("All processes completed.")
