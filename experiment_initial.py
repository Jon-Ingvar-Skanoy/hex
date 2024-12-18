# experiment_initial.py

import os
import platform
from pathlib import Path
import pickle
import numpy as np
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from sklearn.metrics import f1_score, precision_score, recall_score
import time
import argparse

from src.helper_functions import *


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def main(board_size, exp_name, sample_size, open_position, moves_before, message_size, message_bits, double_hashing, epochs, number_of_clauses, T, s, q, depth):

    # "dh2": "double-hash", "dh1": "single-hash?" or "no-hash"
    dh_string = "dh2" if double_hashing else "dh1"

    filename = f'{board_size}x{board_size}_{sample_size}_{open_position}_{moves_before}_{message_size}_{message_bits}_{dh_string}'
    
    machine_name, os_name, user = get_machine_info()
    paths = get_paths(machine_name, os_name, user)
    
    with open(paths['graphs'] / f'{filename}.pkl', 'rb') as f:
        graphs_train, graphs_test, _, Y_train, _, Y_test = pickle.load(f)

    max_included_literals = None
    boost_true_positive_feedback = 1
    number_of_state_bits = 8
    grid = (16*13, 1, 1)
    block = (128, 1, 1)

    acc_test_list = []
    acc_train_list = []
    f1_test_list = []
    precition_test_list = []
    recall_test_list = []

    print("Board Size: %d" % board_size, flush=True)

    for run in range(5):
        print(f"    Run: {run}", flush=True)
        start_time = time.time()
        tm = MultiClassGraphTsetlinMachine(
            number_of_clauses=number_of_clauses,
            T=T,
            s=s,
            q=q,
            depth=depth,
            max_included_literals=max_included_literals,
            boost_true_positive_feedback=boost_true_positive_feedback,
            number_of_state_bits=number_of_state_bits,
            message_size=message_size,
            message_bits=message_bits,
            double_hashing=double_hashing,
            grid=grid,
            block=block
        )

        # Train for epochs
        for i in range(epochs):
            tm.fit(graphs_train, Y_train, epochs=1, incremental=True)

            acc_train = 100 * (tm.predict(graphs_train) == Y_train).mean()

            prediction = tm.predict(graphs_test)
            acc_test = 100 * (prediction == Y_test).mean()
            f1_test = f1_score(Y_test, prediction, average='binary')
        
            print("        Epoch: {:>3} | Train Acc: {:>7.4f} | Test Acc: {:>7.4f} | F1: {:>5.4f} ".format(i, acc_train, acc_test, f1_test), flush=True)

        prediction = tm.predict(graphs_test)

        # Calculate metrics
        acc_train = 100 * (tm.predict(graphs_train) == Y_train).mean()
        acc_test = 100 * (prediction == Y_test).mean()
        f1_test = f1_score(Y_test, prediction, average='binary')
        precision_test = precision_score(Y_test, prediction, average='binary')
        recall_test = recall_score(Y_test, prediction, average='binary')
        
        acc_train_list.append(acc_train)
        acc_test_list.append(acc_test)
        f1_test_list.append(f1_test)
        precition_test_list.append(precision_test)
        recall_test_list.append(recall_test)

        elapsed_time = time.time() - start_time

        param_str = f"b{board_size}_samp{sample_size}_c{number_of_clauses}_T{T}_s{s}_q{q}_d{depth}"
        param_filename = f"{param_str}_{dh_string}"

        # Write results to file
        with open(f"results/experiments/{exp_name}_{param_filename}.txt", "a") as f:
            f.write("Run: %d, Train Accuracy: %.2f, Test Accuracy: %.2f, F1: %.2f, Precision: %.2f, Recall: %.2f, Elapsed Time: %.2f\n" % (
                run, acc_train, acc_test, f1_test, precision_test, recall_test, elapsed_time
            ))

        print(f"    Run: {run}, Train Accuracy: {acc_train}, Test Accuracy: {acc_test}, F1: {f1_test}, Precision: {precision_test}, Recall: {recall_test}", flush=True)

    # Print summary
    print("Test Accuracy Mean: %.2f, Test Accuracy Std: %.2f" % (np.mean(acc_test_list), np.std(acc_test_list)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments for Graph Tsetlin Machine")

    # Add arguments
    parser.add_argument("--board_size", type=int, required=True, help="The size of the board (e.g., 4, 5, 6).")
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name (default: '').")
    parser.add_argument("--sample_size", type=int, default=1000, help="Number of samples to use (e.g., 1000).")
    parser.add_argument("--open_position", type=int, default=40, help="Percentage of open positions (default: 40).")
    parser.add_argument("--moves_before", type=int, default=0, help="Moves before win (default: 0).")
    parser.add_argument("--message_size", type=int, default=128, help="Hypervector size (e.g., 128).")
    parser.add_argument("--message_bits", type=int, default=2, help="Hypervector bits (e.g., 2).")
    parser.add_argument("--double_hashing", type=str2bool, default=False, help="Use double hashing (default: False).")
    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs to train (default: 150).")
    parser.add_argument("--number_of_clauses", type=int, required=True, help="Number of clauses (e.g., 1000).")
    parser.add_argument("--T", type=float, required=True, help="Threshold (T) value.")
    parser.add_argument("--s", type=float, required=True, help="Specificity (s) parameter.")
    parser.add_argument("--q", type=float, required=True, help="q parameter.")
    parser.add_argument("--depth", type=int, required=True, help="Depth of the machine.")

    # Parse arguments
    args = parser.parse_args()

    # Run the main function
    main(
        board_size=args.board_size,
        exp_name=args.exp_name,
        sample_size=args.sample_size,
        open_position=args.open_position,
        moves_before=args.moves_before,
        message_size=args.message_size,
        message_bits=args.message_bits,
        double_hashing=args.double_hashing,
        epochs=args.epochs,
        number_of_clauses=args.number_of_clauses,
        T=args.T,
        s=args.s,
        q=args.q,
        depth=args.depth
    )
