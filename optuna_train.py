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
    number_of_clauses = trial.suggest_int("number_of_clauses", 100, 5000)  
    T = number_of_clauses * trial.suggest_float("T_factor", 0.1, 1.0)
    s = trial.suggest_float("s", 0.1, 1)
    depth = trial.suggest_int("depth", 2, 4)
    max_literals = trial.suggest_int("max_literals", 150, 300)
    message_size = trial.suggest_int("message_size", 500, 800)
    message_bits = trial.suggest_int("message_bits", 1, 6)
    state_bits = trial.suggest_int("state_bits", 32, 64)
    epochs = 70
    
    dataset = trial.suggest_categorical("dataset", ["data.pkl", "data2.pkl", "data3.pkl", "data4.pkl"])
    # data1 = 6x6 1024 6, data2 = 6x6 256 6, data3 = 6x6 256 4, data4 = 6x6 256 2

    with open('data.pkl', 'rb') as f:
        graphs_train, graphs_test, X_train, Y_train, X_test, Y_test = pickle.load(f)


# 
# [number_of_clauses: 3388, T_factor: 0.44279896219762604, s: 0.16282501955746675, max_literals: 222]
# 
# 
    trial

  


    # Initialize the Tsetlin machine with suggested hyperparameters
    tm = MultiClassGraphTsetlinMachine(
        number_of_clauses, T, s, depth=depth, message_size=message_size,
        message_bits=message_bits, number_of_state_bits=state_bits, boost_true_positive_feedback=1,
        max_included_literals=max_literals,
   
        grid=(16*13,1,1),
        block=(128,1,1)


    )

    #  [number_of_clauses: 3388, T_factor: 0.44279896219762604, s: 0.16282501955746675, max_literals: 222]
    trial.set_user_attr("state_bits", state_bits)
    trial.set_user_attr("number_of_clauses", number_of_clauses)
    trial.set_user_attr("T", T)
    trial.set_user_attr("s", s)
    trial.set_user_attr("depth", depth)
    trial.set_user_attr("max_literals", max_literals)
    trial.set_user_attr("message_size", message_size)
    trial.set_user_attr("message_bits", message_bits)
  
   


    best_acc = 0
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
        if result_test >= 100 and result_test < 90:
            break

        # Update best accuracy if new maximum is found
        if result_test > best_acc:
            best_acc = result_test


    return best_acc

# Create study and optimize with user-controlled stopping mechanism
study = optuna.create_study(
    direction="maximize",
    study_name="My_Custom_Study_6x6_varied_vectorsize",  # Name your study
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
