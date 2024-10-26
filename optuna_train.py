import pickle
from GraphTsetlinMachine.graphs import Graphs
import numpy as np
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine, CommonTsetlinMachine
import optuna
from optuna.exceptions import TrialPruned

with open('data.pkl', 'rb') as f:
    graphs_train, graphs_test, X_train, Y_train, X_test, Y_test = pickle.load(f)




def objective(trial):
    # Suggest values for hyperparameters
    number_of_clauses = trial.suggest_int('number_of_clauses', 5000, 60000, step=1000)
    T = number_of_clauses * trial.suggest_float('T_factor', 0.5, 2)
    s = trial.suggest_float('s', 0.0, 5.0)
    depth = trial.suggest_int('depth', 4, 6, step=1)
    max_literals = trial.suggest_int('max_literals', 10, 200, step=1)
    message_size = 32
    message_bits = 2
  
    epochs = 20

    # Initialize the Tsetlin machine with suggested hyperparameters
    tm = MultiClassGraphTsetlinMachine(
        number_of_clauses, T, s, depth=depth, message_size=message_size,
        message_bits=message_bits, number_of_state_bits=8, boost_true_positive_feedback=1,
        max_included_literals=max_literals
    )

    for i in range(epochs):
        # Train the model
        tm.fit(graphs_train, Y_train, epochs=1, incremental=True)

        # Compute accuracy on test and training data
        result_test = 100 * (tm.predict(graphs_test) == Y_test).mean()
    #    result_train = 100 * (tm.predict(graphs_train) == Y_train).mean()

        # Report intermediate result and prune trial if not promising
        trial.report(result_test, i)

        if trial.should_prune():
            raise TrialPruned()

        # Early stopping conditions
        if result_test >= 100 or result_test < 65:
            break

    return result_test

# Create study and optimize with user-controlled stopping mechanism
study = optuna.create_study(
    direction="maximize",
    study_name="My_Custom_Study_5x5_2",  # Name your study
    storage="sqlite:///my_study.db",  # Save to SQLite database
    load_if_exists=True,  # If a study with this name already exists, load it
    pruner=optuna.pruners.MedianPruner()
) 

try:
    # This allows you to stop the optimization process manually
    study.optimize(objective, n_trials=100)
except KeyboardInterrupt:
    print("Optimization interrupted!")
    # Optionally, display the current best result when stopped
    print(f"Best result so far: {study.best_params}")
    # {'number_of_clauses': 100000, 'T_factor': 0.6520892621348986, 's': 1.9616266603257886, 'depth': 9}
