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
def generate_latex_table(base_path, experiment_name, include_params=None, decimals=2, column_decimals=None, sort_by=None, ascending=True, filters=None, rename_columns=None):
    results = []
    for file in os.listdir(base_path):
        if file.startswith(experiment_name.lower()) and file.endswith(".txt"):
            filepath = os.path.join(base_path, file)
            params = extract_params_from_filename(file)
            
            if filters:
                if not all(params.get(k) == v for k, v in filters.items()):
                    continue 

            metrics = process_file(filepath)
            if metrics:
                summary = compute_mean_std(metrics, decimals=decimals)
                results.append({**params, **summary})

    df = pd.DataFrame(results)

    if column_decimals:
        for column, precision in column_decimals.items():
            if column in df.columns:
                df[column] = df[column].apply(lambda x: f"{x:.{precision}f}" if isinstance(x, (float, int)) else x)

    # Include only specified parameters
    if include_params:
        df = df[include_params]

    # Rename columns if a mapping is provided
    if rename_columns:
        df.rename(columns=rename_columns, inplace=True)

    # Sort the table
    if sort_by:
        df = df.sort_values(by=sort_by, ascending=ascending)

    # LaTeX table generation with custom header and footer
    latex_header = f"""
\\begin{{table}}[ht]  
\\centering 
\\begin{{tabular}}{{{'cll' + 'c' * (len(df.columns) - 3)}}}
\\toprule
{' & '.join(df.columns)} \\\\
\\midrule
"""
    
    latex_footer = f"""
\\bottomrule
\\end{{tabular}}

\\smallskip 
\\caption{{Mean (std) of metrics for experiment: {experiment_name}.}}
\\label{{tab:{experiment_name.replace(' ', '_').lower()}}} 
\\end{{table}}
"""

    # Generate LaTeX table content
    latex_rows = []
    grouped = df.groupby("Size")
    for board_size, group in grouped:
        num_rows = len(group)
        latex_rows.append(f"\\multirow{{{num_rows}}}{{*}}{{{board_size}}}")  # Multirow for Board Size

        # Group consecutive rows by Sample Size
        prev_sample_size = None
        sample_size_rows = []
        for i, row in group.iterrows():
            if prev_sample_size is None or row["Samples"] != prev_sample_size:
                # Process previous Sample Size group
                if sample_size_rows:
                    num_sample_size_rows = len(sample_size_rows)
                    first_row = sample_size_rows[0]
                    latex_rows.append(
                        f" & \\multirow{{{num_sample_size_rows}}}{{*}}{{{first_row['Samples']}}} & "
                        + " & ".join(map(str, first_row.values[2:])) + " \\\\"
                    )
                    for sample_row in sample_size_rows[1:]:
                        latex_rows.append(" & & " + " & ".join(map(str, sample_row.values[2:])) + " \\\\")
                    latex_rows.append(f"\\cmidrule(l){{2-{len(df.columns)}}}")

                sample_size_rows = []

            sample_size_rows.append(row)
            prev_sample_size = row["Samples"]

        # Process the last group of Sample Size
        if sample_size_rows:
            num_sample_size_rows = len(sample_size_rows)
            first_row = sample_size_rows[0]
            latex_rows.append(
                f" & \\multirow{{{num_sample_size_rows}}}{{*}}{{{first_row['Samples']}}} & "
                + " & ".join(map(str, first_row.values[2:])) + " \\\\"
            )
            for sample_row in sample_size_rows[1:]:
                latex_rows.append(" & & " + " & ".join(map(str, sample_row.values[2:])) + " \\\\")

        latex_rows.append("\\midrule")  # Midline between Board Sizes

    # Combine header, rows, and footer
    latex_content = "\n".join(latex_rows)
    full_latex_table = f"{latex_header}{latex_content}{latex_footer}"

    return full_latex_table

# Example usage
base_path = "results/experiments"
experiment_name = "Baseline"
include_params = ["Board Size", "Sample Size", "Clauses", "Train Accuracy", "Test Accuracy", "F1"]
decimals = 2
column_decimals = {"T": 0, "s": 1, "q": 1}  # Specify decimal precision for specific columns
ascending = [True, True, True]  # Ascending order for each column
filters = {"s": 1.0}  # Include only rows where s=1.0 and T=800
rename_columns = {
    "Board Size": "Size",
    "Sample Size": "Samples",
    "Train Accuracy": "Train Acc",
    "Test Accuracy": "Test Acc",
    "Elapsed Time": "Time (s)"
}
sort_by = ["Size", "Samples", "Clauses"]  # Sort by multiple columns

# Generate the LaTeX table
latex_table = generate_latex_table(base_path, experiment_name, include_params, decimals, column_decimals, sort_by, ascending, filters, rename_columns)

# Output the table
print(latex_table)
