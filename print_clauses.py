import time
import pickle
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine, CommonTsetlinMachine

with open('11x11__40_0.pkl', 'rb') as f:
    graphs_train, graphs_test, X_train, Y_train, X_test, Y_test = pickle.load(f)

number_of_clauses = 1000
T = number_of_clauses*1.5971422963103452
depth = 2
s = 1
message_size = 64
message_bits = 3

number_of_states = 200
epochs = 5
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

for i in range(epochs):
    # Train the model
        tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
            # Optional: print intermediate test accuracy
        result_test = 100 * (tm.predict(graphs_test) == Y_test).mean()
        print(result_test)
    


     

   

print("st")
weights = tm.get_state()[1].reshape(2, -1)

output_file = "output_results.txt"

with open(output_file, "w") as f:
    f.write(f"Symbol hypervectors: \n{graphs_train.hypervectors}\n\n")
    f.write(f"Clause hypervectors: \n{tm.hypervectors=}\n\n")

    f.write("Clause in Hyperliterals format:\n")
    for clause in range(tm.number_of_clauses):
        f.write(f"Clause {clause} [{weights[0, clause]:>4d} {weights[1, clause]:>4d}]: ")
        f.write(" ".join(
            str(int(tm.ta_action(depth=0, clause=clause, ta=i)))
            for i in range(graphs_train.hypervector_size * 2)
        ))
        f.write("\n")

    f.write("\nMessages as hypervectors:\n")
    for clause in range(tm.number_of_clauses):
        f.write(f"Clause {clause} [{weights[0, clause]:>4d} {weights[1, clause]:>4d}]: ")
        f.write(" ".join(
            str(int(tm.ta_action(depth=1, clause=clause, ta=i)))
            for i in range(tm.message_size * 2)
        ))
        f.write("\n")

    clause_literals = tm.get_clause_literals(graphs_train.hypervectors)
    message_clauses = tm.get_messages(1, len(graphs_train.edge_type_id))
    num_symbols = len(graphs_train.symbol_id)
    symbol_dict = {v: k for k, v in graphs_train.symbol_id.items()}

    f.write("\nActual clauses:\n")
    for clause in range(tm.number_of_clauses):
        f.write(f"Clause {clause} [{weights[0, clause]:>4d} {weights[1, clause]:>4d}]: ")
        for literal in range(num_symbols):
            if clause_literals[clause, literal] > 0:
                f.write(f"{clause_literals[clause, literal]}{symbol_dict[literal]} ")
            if clause_literals[clause, literal + num_symbols] > 0:
                f.write(f"~{clause_literals[clause, literal + num_symbols]}{symbol_dict[literal]} ")
        f.write("\n")

    for edge_type in range(len(graphs_train.edge_type_id)):
        f.write(f"\nActual Messages for {edge_type=}:")
        for msg in range(tm.number_of_clauses):
            f.write(f"\nMessage {msg} : ")
            for clause in range(tm.number_of_clauses):
                if message_clauses[edge_type, msg, clause] == 1:
                    f.write(f"C:{clause}( ")
                    for literal in range(num_symbols):
                        if clause_literals[clause, literal] > 0:
                            f.write(f"{clause_literals[clause, literal]}{symbol_dict[literal]} ")
                        if clause_literals[clause, literal + num_symbols] > 0:
                            f.write(f"~{clause_literals[clause, literal + num_symbols]}{symbol_dict[literal]} ")
                    f.write(") ")

                if message_clauses[edge_type, msg, tm.number_of_clauses + clause] == 1:
                    f.write(f"~C:{clause}( ")
                    for literal in range(num_symbols):
                        if clause_literals[clause, literal] > 0:
                            f.write(f"{clause_literals[clause, literal]}{symbol_dict[literal]} ")
                        if clause_literals[clause, literal + num_symbols] > 0:
                            f.write(f"~{clause_literals[clause, literal + num_symbols]}{symbol_dict[literal]} ")
                    f.write(") ")
            f.write("\n")

print(f"Output written to {output_file}")


