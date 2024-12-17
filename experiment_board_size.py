import time
import pickle
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine, CommonTsetlinMachine

# Hyperparameters
number_of_clauses = 10000
T = number_of_clauses * 1.5971422963103452
depth = 1
s = 1
message_size = 64
message_bits = 3
max_included_literals = 59
number_of_states = 200
epochs = 20
state_bits = 8
max_literals = 20

# Iterate through the parameter combinations
for size in [5, 7, 9, 11, 13]:  # First parameter
    for param in [0, 2, 5]:  # Second parameter
        file_name = f"{size}x{size}__40_{param}.pkl"  # File name format
        try:
            with open(file_name, 'rb') as f:
                graphs_train, graphs_test, X_train, Y_train, X_test, Y_test = pickle.load(f)
        except FileNotFoundError:
            print(f"File {file_name} not found. Skipping.")
            continue

        total_samples = X_train.shape[0] + X_test.shape[0]  # Total number of samples
        print(f"Processing file: {file_name}, Total Samples: {total_samples}")
        
        ac_test_list = []
        ac_train_list = []
        f1_test_list = []
        precision_test_list = []
        recall_test_list = []
        runtime_list = []

        for run in range(5):  # Perform multiple runs
            print(f"File: {file_name}, Run: {run + 1}")

            # Initialize the Tsetlin machine
            tm = MultiClassGraphTsetlinMachine(
                number_of_clauses, T, s, depth=depth, message_size=message_size,
                message_bits=message_bits, number_of_state_bits=state_bits, boost_true_positive_feedback=1,
                max_included_literals=max_literals,
                grid=(16 * 13, 1, 1),
                block=(128, 1, 1)
            )

            start_time = time.time()
            for i in range(epochs):
                # Train the model
                tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
                # Optional: print intermediate test accuracy
                result_test = 100 * (tm.predict(graphs_test) == Y_test).mean()
                print(result_test)
            end_time = time.time()

            # Predict and evaluate
            prediction = tm.predict(graphs_test)

            ac_test = 100 * (prediction == Y_test).mean()
            f1_test = f1_score(Y_test, prediction, average='binary')
            precision_test = precision_score(Y_test, prediction, average='binary')
            recall_test = recall_score(Y_test, prediction, average='binary')
            ac_train = 100 * (tm.predict(graphs_train) == Y_train).mean()

            # Append results
            ac_test_list.append(ac_test)
            ac_train_list.append(ac_train)
            f1_test_list.append(f1_test)
            precision_test_list.append(precision_test)
            recall_test_list.append(recall_test)
            runtime_list.append(end_time - start_time)

        # Write results to file
        with open("results.txt", "a") as f:
            f.write(f"File: {file_name}, Total Samples: {total_samples}, Depth: {depth}, "
                    f"Test Accuracy Mean: {np.mean(ac_test_list):.4f} (std: {np.std(ac_test_list):.4f}), "
                    f"F1 Mean: {np.mean(f1_test_list):.4f} (std: {np.std(f1_test_list):.4f}), "
                    f"Precision Mean: {np.mean(precision_test_list):.4f} (std: {np.std(precision_test_list):.4f}), "
                    f"Recall Mean: {np.mean(recall_test_list):.4f} (std: {np.std(recall_test_list):.4f}), "
                    f"Train Accuracy Mean: {np.mean(ac_train_list):.4f} (std: {np.std(ac_train_list):.4f}), "
                    f"Time Mean: {np.mean(runtime_list):.4f} (std: {np.std(runtime_list):.4f})\n")

        # Print summary
        print(f"File: {file_name}, Total Samples: {total_samples}, Depth: {depth}, "
              f"Test Accuracy Mean: {np.mean(ac_test_list):.2f}, Test Accuracy Std: {np.std(ac_test_list):.2f}")
