from collections import defaultdict
import re
from matplotlib import pyplot as plt
import numpy as np

input_file = "output_results.txt"

def count_literals(file_path):
    literal_counts = defaultdict(int)
    pattern = re.compile(r"(~?)[0-9\.]+([A-Z_]+\d*)")  # Match ~0.5RED, 1.0ROW_5, etc., keep ~ and suffix

    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("Clause"):
                literals = line.split()[3:]  # Extract literals after the clause header
                for literal in literals:
                    literal = literal.strip("()")
                    match = pattern.match(literal)
                    if match:
                        cleaned_literal = match.group(1) + match.group(2)  # Keep ~ if present and suffix
                        literal_counts[cleaned_literal] += 1
    return literal_counts

# Count literals from the file
literal_counts = count_literals(input_file)

# Print the results
print("Literal counts:")
for literal, count in sorted(literal_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{literal}: {count}")



def count_combinations(file_path):
    combinations = defaultdict(int)
    row_pattern = re.compile(r"(~?)[0-9\.]+ROW_(\d+)")  # Match ROW combinations
    col_pattern = re.compile(r"(~?)[0-9\.]+COL_(\d+)")  # Match COL combinations

    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("Clause"):
                row_indices = set()
                col_indices = set()
                literals = line.split()[3:]  # Extract literals after the clause header
                for literal in literals:
                    literal = literal.strip("()")
                    row_match = row_pattern.match(literal)
                    col_match = col_pattern.match(literal)
                    if row_match:
                        negation = row_match.group(1)
                        row_index = int(row_match.group(2))
                        row_indices.add((negation, row_index))
                    if col_match:
                        negation = col_match.group(1)
                        col_index = int(col_match.group(2))
                        col_indices.add((negation, col_index))
                # Update combinations for each row-col pair
                for row in row_indices:
                    for col in col_indices:
                        key = f"{row[0]}ROW_{row[1]}:{col[0]}COL_{col[1]}"
                        combinations[key] += 1
    return combinations

def plot_heatmap(combinations, only_negated=False):
    # Initialize heatmap matrix (11x11 for ROW 0-10 and COL 0-10)
    heatmap = np.zeros((11, 11))

    for key, count in combinations.items():
        if only_negated and "~" not in key:
            continue
        if not only_negated and "~" in key:
            continue

        row_match = re.search(r"ROW_(\d+)", key)
        col_match = re.search(r"COL_(\d+)", key)
        if row_match and col_match:
            row = int(row_match.group(1))
            col = int(col_match.group(1))
            if 0 <= row <= 10 and 0 <= col <= 10:
                heatmap[row, col] += count

    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap, cmap="Blues", origin="upper")
    plt.xticks(np.arange(11), [f"COL_{i}" for i in range(11)], rotation=45)
    plt.yticks(np.arange(11), [f"ROW_{i}" for i in range(11)])
    plt.title("Heatmap of ROW-COL Combinations" + (" (Only ~)" if only_negated else ""))
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.show()

# Count combinations from the file
combinations = count_combinations(input_file)

# Print the results
print("Combination counts:")
for key, count in sorted(combinations.items(), key=lambda x: x[1], reverse=True):
    print(f"{key}: {count}")

# Plot two heatmaps: one for ~ combinations and one for non-~ combinations
plot_heatmap(combinations, only_negated=True)
plot_heatmap(combinations, only_negated=False)

