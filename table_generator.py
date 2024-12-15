import os
import pandas as pd
import re

# Function to extract parameters from filename
def extract_params_from_filename(filename):
    match = re.search(r"b(\d+)_samp(\d+)_c(\d+)_T([\d.]+)_s([\d.]+)_q([\d.]+)_d(\d+)_dh(\d+)", filename)
    if match:
        return {
            "Board Size": int(match.group(1)),
            "Sample Size": int(match.group(2)),
            "Clauses": int(match.group(3)),
            "T": float(match.group(4)),
            "s": float(match.group(5)),
            "q": float(match.group(6)),
            "Depth": int(match.group(7)),
            "Hashing": f"dh{match.group(8)}"
        }
    else:
        print(f"File: {filename} is not a match")
    return {}

# Function to process a file and extract results
def process_file(filepath):
    metrics = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith("Run:"):
                # Extract metrics from the line
                match = re.search(r"Train Accuracy: ([\d.]+), Test Accuracy: ([\d.]+), F1: ([\d.]+), "
                                  r"Precision: ([\d.]+), Recall: ([\d.]+), Elapsed Time: ([\d.]+)", line)
                if match:
                    metrics.append({
                        "Train Accuracy": float(match.group(1)),
                        "Test Accuracy": float(match.group(2)),
                        "F1": float(match.group(3)),
                        "Precision": float(match.group(4)),
                        "Recall": float(match.group(5)),
                        "Elapsed Time": float(match.group(6)),
                    })
    return metrics

# Function to compute mean and std for a list of metrics
def compute_mean_std(metrics, decimals=2):
    df = pd.DataFrame(metrics)
    summary = df.mean().round(decimals).to_dict()
    std_dev = df.std().round(decimals).to_dict()
    return {metric: (f"{summary[metric]:.{decimals}f} ({std_dev[metric]:.{decimals}f})") for metric in summary}

# Main function to generate LaTeX table
def generate_latex_table(base_path, experiment_name, include_params=None, decimals=2, column_decimals=None, sort_by=None, ascending=True):
    # Process all files and collect results
    results = []
    for file in os.listdir(base_path):
        if file.endswith(".txt"):
            filepath = os.path.join(base_path, file)
            params = extract_params_from_filename(file)
            metrics = process_file(filepath)
            if metrics:
                summary = compute_mean_std(metrics, decimals=decimals)
                results.append({**params, **summary})

    # Convert results to a DataFrame
    df = pd.DataFrame(results)

    # Apply specific formatting for columns
    if column_decimals:
        for column, precision in column_decimals.items():
            if column in df.columns:
                df[column] = df[column].apply(lambda x: f"{x:.{precision}f}" if isinstance(x, (float, int)) else x)

    # Include only specified parameters
    if include_params:
        df = df[include_params]

    # Sort the table
    if sort_by:
        df = df.sort_values(by=sort_by, ascending=ascending)

    # LaTeX table generation with custom header and footer
    latex_header = f"""
\\begin{{table*}}[ht]  
\\centering 
"""
    latex_footer = f"""
\\smallskip 
\\caption{{Mean (standard deviation) of metrics for experiment: {experiment_name}.}}
\\label{{tab:{experiment_name.replace(' ', '_')}}} 
\\end{{table*}}
"""

    # Generate LaTeX table content
    latex_content = df.to_latex(index=False, column_format='c' * len(df.columns), escape=False)

    # Combine header, content, and footer
    full_latex_table = f"{latex_header}{latex_content}{latex_footer}"

    return full_latex_table


# Example usage
base_path = "results/experiments"
experiment_name = "Baseline Experiment"
include_params = ["Board Size", "Sample Size", "Clauses", "T", "s", "q", "Train Accuracy", "Test Accuracy", "F1"]
decimals = 4
column_decimals = {"T": 0, "s": 1, "q": 1}  # Specify decimal precision for specific columns
sort_by = ["Board Size", "Sample Size", "Clauses"]  # Sort by multiple columns
ascending = [True, True, True]  # Ascending order for each column

# Generate the LaTeX table
latex_table = generate_latex_table(base_path, experiment_name, include_params, decimals, column_decimals, sort_by, ascending)

# Output the table
print(latex_table)
