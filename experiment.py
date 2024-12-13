import pickle
from GraphTsetlinMachine.graphs import Graphs
import numpy as np
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine, CommonTsetlinMachine
import optuna
from optuna.exceptions import TrialPruned
from sklearn.metrics import f1_score, precision_score, recall_score

with open('data.pkl', 'rb') as f:
    graphs_train, graphs_test, X_train, Y_train, X_test, Y_test = pickle.load(f)





number_of_clauses = 3500
T = number_of_clauses*1.5971422963103452
depth = 2
s = 1
message_size = 64
message_bits = 3
max_included_literals = 59
number_of_states = 200
epochs = 20
state_bits = 8
max_literals = 20




  


    # Initialize the Tsetlin machine with suggested hyperparameters
tm = MultiClassGraphTsetlinMachine(
        number_of_clauses, T, s, depth=depth, message_size=message_size,
        message_bits=message_bits, number_of_state_bits=state_bits, boost_true_positive_feedback=1,
        max_included_literals=max_literals,
   
        grid=(16*13,1,1),
        block=(128,1,1)


    )

    
for variant in [1, 2, 3, 5,10]:   

    depth = variant 

    ac_test_list = []
    ac_train_list = []
    f1_test_list = []
    precition_test_list = []
    recall_test_list = []


    for run in range(5):

        print("Depth: %d, Run: %d" % (depth, run))

        # Initialize the Tsetlin machine with suggested hyperparameters
        tm = MultiClassGraphTsetlinMachine(
            number_of_clauses, T, s, depth=depth, message_size=message_size,
            message_bits=message_bits, number_of_state_bits=state_bits, boost_true_positive_feedback=1,
            max_included_literals=max_literals,
            grid=(16*13,1,1),
            block=(128,1,1)
        )

        


        
    



        for i in range(epochs):
            # Train the model
            tm.fit(graphs_train, Y_train, epochs=1, incremental=True)

                # Compute accuracy on test and training data
            result_test = 100 * (tm.predict(graphs_test) == Y_test).mean()

            print(result_test)
            #    result_train = 100 * (tm.predict(graphs_train) == Y_train).mean()

                # Report intermediate result and prune trial if not promising

        
        prediciton = tm.predict(graphs_test)

        ac_test = 100 * (prediciton == Y_test).mean()
        f1_test = f1_score(Y_test, prediciton, average='binary')
        precision_test = precision_score(Y_test, prediciton, average='binary')
        recall_test = recall_score(Y_test, prediciton, average='binary')
        ac_train = 100 * (tm.predict(graphs_train) == Y_train).mean()

        ac_test_list.append(ac_test)
        ac_train_list.append(ac_train)
        f1_test_list.append(f1_test)
        precition_test_list.append(precision_test)
        recall_test_list.append(recall_test)

       with open("results.txt", "a") as f:
            f.write("Depth: %d, Run: %d, Train Accuracy: %.2f, Test Accuracy: %.2f, F1: %.2f, Precision: %.2f, Recall: %.2f\n" % (depth, run, ac_train, ac_test, f1_test, precision_test, recall_test))

    print("Depth: %d, Test Accuracy Mean: %.2f, Test Accuracy Std: %.2f" % (depth, np.mean(ac_test_list), np.std(ac_test_list)))


     

