import os
import pandas as pd
import re
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
import math

# Function to extract parameters from filename
def extract_params_from_filename(filename):
    match = re.search(r"b(\d+)_samp(\d+)_c(\d+)_T([\d.]+)_s([\d.]+)_q([\d.]+)_d(\d+)_ms(\d+)_mb(\d+)_op(\d+)_dh(\d+)", filename)
    if match:
        return {
            "Board Size": int(match.group(1)),
            "Sample Size": int(match.group(2)),
            "Clauses": int(match.group(3)),
            "T": float(match.group(4)),
            "s": float(match.group(5)),
            "q": float(match.group(6)),
            "Depth": int(match.group(7)),
            "Message Size": int(match.group(8)),
            "Message Bits": int(match.group(9)),
            "Open Position": int(match.group(10)),
            "Hashing": f"dh{match.group(11)}"
        }
    else:
        print(f"File: {filename} is not a match")
    return {}

def extract_params_from_log_filename(filename):
    match = re.search(r"b(\d+)_.*_ms(\d+)_mb(\d+)_.*", filename)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return None, None, None

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

def process_log_file(filepath):
    data = []
    run, epoch = None, 0

    with open(filepath, 'r') as file:
        for line in file:
            if line.startswith("Run:"):
                run = int(re.search(r"Run: (\d+)", line).group(1))
                epoch = 0
            elif "Epoch:" in line and "Test Accuracy" in line:
                match = re.search(r"Epoch: (\d+), Test Accuracy: ([\d.]+)", line)
                if match:
                    epoch = int(match.group(1))
                    test_acc = float(match.group(2))
                    data.append({
                        "Run": run,
                        "Epoch": epoch,
                        "Test Accuracy": test_acc
                    })
            elif "Run:" in line and "Train Accuracy" in line:
                match = re.search(r"Train Accuracy: ([\d.]+), Test Accuracy: ([\d.]+)", line)
                if match:
                    train_acc = float(match.group(1))
                    test_acc = float(match.group(2))
                    data.append({
                        "Run": run,
                        "Epoch": epoch + 1,  # Add a pseudo-epoch for the final stats
                        "Train Accuracy": train_acc,
                        "Test Accuracy": test_acc
                    })
    return pd.DataFrame(data)

# Function to compute mean and std for a list of metrics
def compute_mean_std(metrics, decimals=2):
    df = pd.DataFrame(metrics)
    summary = df.mean().round(decimals).to_dict()
    std_dev = df.std().round(decimals).to_dict()
    return {metric: (f"{summary[metric]:.{decimals}f} ({std_dev[metric]:.{decimals}f})") for metric in summary}

def compute_mean_std2(metrics):
    df = pd.DataFrame(metrics)
    mean_values = df.mean().to_dict()
    std_values = df.std().to_dict()
    return mean_values, std_values

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
& \\multicolumn{{{(len(df.columns)-1)}}}{{c}}{{{experiment_name}\\tablefootnote{{Exp. "{experiment_name}":}}}} \\\\
\\cmidrule(l){{2-{(len(df.columns))}}}
{' & '.join(df.columns)} \\\\
\\midrule
"""
    
    latex_footer = f"""
\\bottomrule
\\end{{tabular}}
\\smallskip 
\\caption[Result {experiment_name}]{{Mean (std) of metrics for experiment: {experiment_name}.}}
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

def create_visualizations(df, output_dir, experiment_name):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Colors and sizes for plots
    colors = {2: 'red', 8: 'blue', 16: 'green'}
    sizes = {128: 50, 1024: 100, 8192: 200}

    # Sort board sizes for consistent left-to-right order
    board_sizes = sorted(df["Board Size"].unique())
    num_boards = len(board_sizes)

    # Scatter Plot: Subplots for each board size
    fig, axes = plt.subplots(1, num_boards, figsize=(5 * num_boards, 15), sharey=True)

    if num_boards == 1:  # Single board size, no subplots needed
        axes = [axes]

    for ax, board_size in zip(axes, board_sizes):
        subset = df[df["Board Size"] == board_size]
        for _, row in subset.iterrows():
            ax.scatter(row["Clauses"], row["Test Accuracy Mean"],
                       color=colors[row["Message Bits"]],
                       s=sizes[row["Message Size"]],
                       alpha=0.7)
        ax.set_title(f"Board Size: {board_size}")
        ax.set_xlabel("Clauses")
        ax.set_ylabel("Test Accuracy (Mean)")

        # Create legends for Message Bits and Message Size
        color_legend = [plt.Line2D([0], [0], color=color, marker='o', markersize=8, label=f"Bits {mb}")
                        for mb, color in colors.items()]
        size_legend = [plt.Line2D([0], [0], color='gray', marker='o', markersize=size / 20, label=f"Size {ms}")
                       for ms, size in sizes.items()]

        ax.legend(handles=color_legend + size_legend, title="Legend", loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{experiment_name}_scatter_test_accuracy_by_board.png"))
    plt.close()

    # Scatter Plot: Combined board sizes
    plt.figure(figsize=(12, 8))
    for board_size in sorted(df["Board Size"].unique()):
        subset = df[df["Board Size"] == board_size]
        plt.scatter(subset["Test Accuracy Mean"], subset["Clauses"],
                    label=f"Board {board_size}",
                    c=subset["Message Bits"].map(colors),
                    s=subset["Message Size"].map(sizes),
                    alpha=0.7)
    plt.xlabel("Test Accuracy (Mean)")
    plt.ylabel("Clauses")
    plt.title("Test Accuracy vs Clauses (Flipped Axes)")
    plt.legend(title="Board Size", loc="best")
    plt.savefig(os.path.join(output_dir, f"{experiment_name}_scatter_test_accuracy_combined.png"))
    plt.close()

    # Box Plot: Expanded to include Clauses and Board Size
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="Message Bits", y="Test Accuracy Mean", hue="Board Size", data=df, palette="Set2")
    plt.title("Distribution of Test Accuracy by Message Bits and Board Size")
    plt.xlabel("Message Bits")
    plt.ylabel("Test Accuracy (Mean)")
    plt.legend(title="Board Size", loc="lower right")
    plt.savefig(os.path.join(output_dir, f"{experiment_name}_boxplot_test_accuracy_expanded.png"))
    plt.close()

    # Heatmap: Test Accuracy for Message Bits and HV Size
    pivot_table = df.pivot_table(index="Message Size", columns="Message Bits", values="Test Accuracy Mean", aggfunc="mean")
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_table, annot=True, cmap="Blues", fmt=".2f")
    plt.title("Test Accuracy Heatmap: Message Size vs Message Bits")
    plt.savefig(os.path.join(output_dir, f"{experiment_name}_heatmap_test_accuracy.png"))
    plt.close()

    # Line Plot: Test Accuracy vs Clauses for each Board Size
    plt.figure(figsize=(16, 10))
    for board_size in board_sizes:
        subset = df[df["Board Size"] == board_size]
        for hv_size in subset["Message Size"].unique():
            sub_subset = subset[subset["Message Size"] == hv_size]
            plt.plot(sub_subset["Clauses"], sub_subset["Test Accuracy Mean"],
                     marker='o', label=f"Board {board_size}, HV Size {hv_size}")
    plt.xlabel("Clauses")
    plt.ylabel("Test Accuracy (Mean)")
    plt.title("Test Accuracy vs Clauses by Board Size and HV Size")
    plt.legend(title="Legend", loc="best")
    plt.savefig(os.path.join(output_dir, f"{experiment_name}_lineplot_test_accuracy_by_board.png"))
    plt.close()

    # Bar Plot: Average Test Accuracy by Clauses, Board Size, and Message Size
    grouped_data = df.groupby(["Clauses", "Board Size", "Message Size"])["Test Accuracy Mean"].mean().reset_index()
    plt.figure(figsize=(14, 8))
    sns.barplot(x="Clauses", y="Test Accuracy Mean", hue="Board Size", data=grouped_data, palette="muted")
    plt.title("Average Test Accuracy by Clauses, Board Size, and Message Size")
    plt.xlabel("Clauses")
    plt.ylabel("Average Test Accuracy")
    plt.legend(title="Board Size", loc="best")
    plt.savefig(os.path.join(output_dir, f"{experiment_name}_barplot_test_accuracy_by_board.png"))
    plt.close()

    # Bar Plot: Faceted by Message Bits
    # plt.figure(figsize=(12, 8))
    # sns.barplot(x="Clauses", y="Test Accuracy Mean", hue="Board Size", data=df, palette="muted")
    # plt.title("Average Test Accuracy by Clauses, Board Size, and Message Size")
    # plt.xlabel("Clauses")
    # plt.ylabel("Average Test Accuracy")
    # plt.legend(title="Board Size", loc="best")
    # plt.savefig(os.path.join(output_dir, f"{experiment_name}_barplot_test_accuracy_by_board.png"))
    # plt.close()

    # 3D Scatter Plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    for board_size in sorted(df["Board Size"].unique()):
        subset = df[df["Board Size"] == board_size]
        ax.scatter(subset["Clauses"], subset["Message Bits"], subset["Test Accuracy Mean"],
                   label=f"Board {board_size}",
                   c=subset["Message Bits"].map(colors),
                   s=subset["Message Size"].map(sizes),
                   alpha=0.7)
    ax.set_xlabel("Clauses")
    ax.set_ylabel("Message Bits")
    ax.set_zlabel("Test Accuracy (Mean)")
    ax.set_title("3D Scatter: Clauses, Message Bits, and Test Accuracy")
    ax.legend(title="Board Size", loc="best")
    plt.savefig(os.path.join(output_dir, f"{experiment_name}_3d_scatter_test_accuracy.png"))
    plt.close()

    # 3D Surface Plot: Ensure unique combinations for pivot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    for board_size in sorted(df["Board Size"].unique()):
        subset = df[df["Board Size"] == board_size]
        
        # Group by 'Clauses' and 'Message Bits' to ensure unique entries
        grouped = subset.groupby(["Clauses", "Message Bits"]).agg({"Test Accuracy Mean": "mean"}).reset_index()
        
        # Create meshgrid for the surface plot
        X, Y = np.meshgrid(grouped["Clauses"].unique(), grouped["Message Bits"].unique())
        Z = grouped.pivot(index="Message Bits", columns="Clauses", values="Test Accuracy Mean").values
        
        # Handle any missing values (NaN) with zeros or interpolation
        Z = pd.DataFrame(Z).fillna(0).values
        
        ax.plot_surface(X, Y, Z, alpha=0.6, label=f"Board {board_size}")
    
    ax.set_xlabel("Clauses")
    ax.set_ylabel("Message Bits")
    ax.set_zlabel("Test Accuracy (Mean)")
    ax.set_title("3D Surface: Test Accuracy by Clauses and Message Bits")
    plt.savefig(os.path.join(output_dir, f"{experiment_name}_3d_surface_test_accuracy.png"))
    plt.close()

def plot_results(df, output_dir, board_size, message_size, message_bits):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(10, 6))
    sns.lineplot(x="Epoch", y="Test Accuracy", hue="Run", data=df)
    plt.title(f"Test Accuracy over Epochs (Board Size {board_size}, MS {message_size}, MB {message_bits})")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.legend(title="Run")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"lineplot_board{board_size}_ms{message_size}_mb{message_bits}.png"))
    plt.close()

    # Average test accuracy per epoch
    plt.figure(figsize=(10, 6))
    avg_df = df.groupby("Epoch")["Test Accuracy"].mean().reset_index()
    sns.lineplot(x="Epoch", y="Test Accuracy", data=avg_df, color="blue")
    plt.title(f"Average Test Accuracy over Epochs (Board Size {board_size}, MS {message_size}, MB {message_bits})")
    plt.xlabel("Epoch")
    plt.ylabel("Average Test Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"avg_lineplot_board{board_size}_ms{message_size}_mb{message_bits}.png"))
    plt.close()

    # Boxplot for final test accuracy
    final_df = df[df["Epoch"] == df["Epoch"].max()]
    plt.figure(figsize=(8, 6))
    sns.boxplot(y="Test Accuracy", data=final_df)
    plt.title(f"Final Test Accuracy Distribution (Board Size {board_size}, MS {message_size}, MB {message_bits})")
    plt.ylabel("Test Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"boxplot_board{board_size}_ms{message_size}_mb{message_bits}.png"))
    plt.close()

def plot_training_results(logs_dir, output_dir, experiment_name):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Collect all log data
    all_data = []

    for file in os.listdir(logs_dir):
        if file.startswith(experiment_name.lower()) and file.endswith(".log"):
            filepath = os.path.join(logs_dir, file)
            board_size, message_size, message_bits = extract_params_from_log_filename(file)
            if board_size is None:
                print(f"Skipping file with invalid format: {file}")
                continue

            # Process file
            df = process_log_file(filepath)
            if not df.empty:
                df["Board Size"] = board_size
                df["Message Size"] = message_size
                df["Message Bits"] = message_bits
                all_data.append(df)
            else:
                print(f"No valid data in {file}")

    # Combine all data into a single DataFrame
    if not all_data:
        print("No valid data to plot.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)

    # Generate distinct base colors for Message Sizes and Message Bits
    ms_palette = sns.color_palette("husl", n_colors=combined_df["Message Size"].nunique())
    ms_to_color = {ms: ms_palette[i] for i, ms in enumerate(sorted(combined_df["Message Size"].unique()))}

    mb_palette = sns.color_palette("icefire", n_colors=combined_df["Message Bits"].nunique())
    mb_to_color = {mb: mb_palette[i] for i, mb in enumerate(sorted(combined_df["Message Bits"].unique()))}

    # Sort data for consistent plotting
    board_sizes = sorted(combined_df["Board Size"].unique())
    message_sizes = sorted(combined_df["Message Size"].unique())

    # Boxplot: Variability in Accuracy by HV Sizes and HV Bits
    plt.figure(figsize=(12, 8))
    sns.boxplot(
        data=combined_df,
        x="Message Bits",
        y="Test Accuracy",
        hue="Message Size",
        palette=ms_palette
    )
    plt.title("Variability in Test Accuracy by Message Bits and Message Size", fontsize=16)
    plt.xlabel("Message Bits", fontsize=14)
    plt.ylabel("Test Accuracy", fontsize=14)
    plt.legend(title="Message Size", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{experiment_name}_accuracy_variability_boxplot.png"))
    plt.close()





    # Combined plot: 4 rows x 3 columns
    num_rows, num_cols = 4, 3
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 20), sharex=True, sharey=True)
    axes = axes.flatten()

    subplot_idx = 0
    global_handles_labels = {}

    for board_size in board_sizes:
        subset = combined_df[combined_df["Board Size"] == board_size]

        for msg_size in message_sizes:
            ax = axes[subplot_idx]
            ms_subset = subset[subset["Message Size"] == msg_size]

            # Plot for each Message Bits
            for msg_bits, group in ms_subset.groupby("Message Bits"):
                line_color = mb_to_color[msg_bits]
                sns.lineplot(
                    data=group,
                    x="Epoch",
                    y="Test Accuracy",
                    label=f"MB={msg_bits}",
                    color=line_color,
                    ax=ax
                )

            # Add smoothed average accuracy line
            smooth_w = 10
            avg_accuracy = ms_subset.groupby("Epoch")["Test Accuracy"].mean().rolling(window=smooth_w).mean()
            ax.plot(avg_accuracy.index, avg_accuracy.values, linestyle="--", color="black", label=f"Smoothed Avg Acc (window={smooth_w})")

            # Collect handles and labels for the global legend
            handles, labels = ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                global_handles_labels[label] = handle

            ax.set_title(f"MS={msg_size}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(f"Board {board_size}")

            # Disable subplot-specific legends
            ax.legend().set_visible(False)
            #ax.legend(title="Message Bits", loc="lower right")

            # Save individual full-size plot
            plt.figure(figsize=(10, 6))
            for msg_bits, group in ms_subset.groupby("Message Bits"):
                line_color = mb_to_color[msg_bits]
                sns.lineplot(
                    data=group,
                    x="Epoch",
                    y="Test Accuracy",
                    label=f"MB={msg_bits}",
                    color=line_color
                )
            plt.plot(avg_accuracy.index, avg_accuracy.values, linestyle="--", color="black", label="Smoothed Avg Accuracy")
            plt.title(f"Board {board_size}, MS={msg_size}")
            plt.xlabel("Epoch")
            plt.ylabel("Test Accuracy")
            plt.legend(title="Message Bits", loc="lower right")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{experiment_name}_board{board_size}_ms{msg_size}.png"))
            plt.close()

            subplot_idx += 1

    # Hide any unused subplots
    for i in range(subplot_idx, len(axes)):
        fig.delaxes(axes[i])
    
    global_legend_handles = list(global_handles_labels.values())
    global_legend_labels = list(global_handles_labels.keys())

    fig.legend(
        global_legend_handles,
        global_legend_labels,
        title="Message Bits and Avg Accuracy",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.1), 
        ncol=4,
        fontsize='large'
    )

    # Add a global suptitle
    fig.suptitle("Test Accuracy by Board Size, Message Size, and Message Bits", fontsize=24)
    fig.align_xlabels()
    
    #plt.suptitle("Test Accuracy by Board Size, Message Size and Message Bits", fontsize=24)
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])  
    plt.savefig(os.path.join(output_dir, f"{experiment_name}_combined_training_results.png"))
    plt.close()


    # Subplots 20 epochs
    first_20_epochs_df = combined_df[combined_df["Epoch"] <= 20]

    # Combined plot: 4 rows x 3 columns
    num_rows, num_cols = 4, 3
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 20), sharex=True, sharey=True)
    axes = axes.flatten()

    subplot_idx = 0
    global_handles_labels = {}

    for board_size in board_sizes:
        subset = first_20_epochs_df[first_20_epochs_df["Board Size"] == board_size]

        for msg_size in message_sizes:
            ax = axes[subplot_idx]
            ms_subset = subset[subset["Message Size"] == msg_size]

            # Plot for each Message Bits
            for msg_bits, group in ms_subset.groupby("Message Bits"):
                line_color = mb_to_color[msg_bits]
                sns.lineplot(
                    data=group,
                    x="Epoch",
                    y="Test Accuracy",
                    label=f"MB={msg_bits}",
                    color=line_color,
                    ax=ax
                )

            # Add smoothed average accuracy line
            smooth_w = 2
            avg_accuracy = ms_subset.groupby("Epoch")["Test Accuracy"].mean().rolling(window=smooth_w).mean()
            ax.plot(avg_accuracy.index, avg_accuracy.values, linestyle="--", color="black", label=f"Smoothed Avg Acc (window={smooth_w})")


            # Collect handles and labels for the global legend
            handles, labels = ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                global_handles_labels[label] = handle

            ax.set_title(f"MS={msg_size}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(f"Board {board_size}")

            # Disable subplot-specific legends
            ax.legend().set_visible(False)

            plt.figure(figsize=(10, 6))
            for msg_bits, group in ms_subset.groupby("Message Bits"):
                line_color = mb_to_color[msg_bits]
                sns.lineplot(
                    data=group,
                    x="Epoch",
                    y="Test Accuracy",
                    label=f"MB={msg_bits}",
                    color=line_color
                )
            plt.plot(avg_accuracy.index, avg_accuracy.values, linestyle="--", color="black", label=f"Smoothed Avg Acc (window={smooth_w})")
            plt.title(f"Board {board_size}, MS={msg_size}")
            plt.xlabel("Epoch")
            plt.ylabel("Test Accuracy")
            plt.legend(title="Message Bits", loc="lower right")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{experiment_name}_board{board_size}_ms{msg_size}_first_20_epochs.png"))
            plt.close()

            subplot_idx += 1

    # Hide any unused subplots
    for i in range(subplot_idx, len(axes)):
        fig.delaxes(axes[i])
    
    global_legend_handles = list(global_handles_labels.values())
    global_legend_labels = list(global_handles_labels.keys())

    fig.legend(
        global_legend_handles,
        global_legend_labels,
        title="Message Bits and Avg Accuracy",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.1), 
        ncol=4,
        fontsize='large'
    )

    # Add a global suptitle
    fig.suptitle("Test Accuracy by Board Size, Message Size, and Message Bits (First 20 Epochs)", fontsize=24)
    fig.align_xlabels()
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])  
    plt.savefig(os.path.join(output_dir, f"{experiment_name}_first_20_epochs_combined_training_results.png"))
    plt.close()



    # Original Summary Plot
    ms_linestyles = {
        128: 'solid',  
        1024: 'dashed', 
        8192: 'dotted' 
    }

    mb_markers = {2: 'o', 8: 's', 16: 'D'}
    summary_df = combined_df.groupby(["Board Size", "Message Size", "Message Bits", "Epoch"])["Test Accuracy"].mean().reset_index()

    # Create color palette for board sizes
    board_size_palette = sns.color_palette("husl", n_colors=summary_df["Board Size"].nunique())
    board_to_color = {board: board_size_palette[i] for i, board in enumerate(sorted(summary_df["Board Size"].unique()))}

    # Original Summary Plot
    plt.figure(figsize=(15, 8))
    for (board_size, msg_size, msg_bits), group in summary_df.groupby(["Board Size", "Message Size", "Message Bits"]):
        label = f"Board {board_size}, MS={msg_size}, MB={msg_bits}"
        color = board_to_color[board_size]  # Use board size color
        sns.lineplot(
            data=group,
            x="Epoch",
            y="Test Accuracy",
            label=label,
            color=color,
            marker=mb_markers.get(msg_bits, 'o'),  # Use marker based on MB
            linestyle= ms_linestyles.get(msg_size, 'solid'),
            markevery=(board_size+int(math.sqrt(msg_bits)))
        )


    plt.title("Summary: Average Accuracy Across Configurations")
    plt.xlabel("Epoch")
    plt.ylabel("Average Test Accuracy")
    plt.legend(title="Configuration", loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize='small')
    plt.tight_layout(rect=[0, 0, 1, 0.85])
    plt.savefig(os.path.join(output_dir, f"{experiment_name}_summary_plot.png"))
    plt.close()

    # Smoothed Summary Plot
    plt.figure(figsize=(12, 18))  
    for (board_size, msg_size, msg_bits), group in summary_df.groupby(["Board Size", "Message Size", "Message Bits"]):
        label = f"Board {board_size}, MS={msg_size}, MB={msg_bits}"
        color = board_to_color[board_size]  # Use board size color
        smoothed_test_acc = group["Test Accuracy"].rolling(window=10).mean()  # Apply rolling mean for smoothing
        plt.plot(
            group["Epoch"],
            smoothed_test_acc,
            label=label,
            color=color,
            marker=mb_markers.get(msg_bits, 'o'),  # Use marker based on MB
            linestyle=ms_linestyles.get(msg_size, 'solid'),
            markevery=(board_size+int(math.sqrt(msg_bits)))
        )

    plt.title("Smoothed Summary: Average Accuracy Across Configurations", fontsize=24)
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("Average Test Accuracy", fontsize=18)
    plt.legend(title="Configuration", loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize='medium')
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(os.path.join(output_dir, f"{experiment_name}_smoothed_summary_plot.png"))
    plt.close()

    print(f"Combined plots, individual subplots, original summary, and smoothed summary plot saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a LaTeX table from experiment results.")
    parser.add_argument("--experiment_name", required=True, help="Name of the experiment to process.")
    parser.add_argument("--output_path", required=True, help="Path to save the LaTeX table.")

    args = parser.parse_args()
    experiment_name = args.experiment_name
    output_path = args.output_path

    experiments_dir = "results/experiments"
    logs_dir = "logs"
    plot_dir = "results/plots"

    include_params = ["Board Size", "Sample Size", "Clauses",  "Train Accuracy","Test Accuracy",  "F1"]
    decimals = 2
    column_decimals = {"T": 0, "s": 1, "q": 1}
    sort_by = ["Size", "Samples", "Clauses"]
    ascending = [True, True, True]
    filters = {}  # Add filters if needed
    rename_columns = {
        "Board Size": "Size",
        "Sample Size": "Samples",
        # "Message Size": "HV Size",
        # "Message Bits": "HV Bits",
        "Train Accuracy": "Train Acc",
        "Test Accuracy": "Test Acc",
        "Elapsed Time": "Time (s)"
    }

    # Generate the LaTeX table
    latex_table = generate_latex_table(experiments_dir, experiment_name, include_params, decimals, column_decimals, sort_by, ascending, filters, rename_columns)

    # Save the LaTeX table to the specified file
    with open(output_path, "w") as f:
        f.write(latex_table)

    print(f"LaTeX table saved to {output_path}.")



    # # Create visualizations
    # results = []
    # for file in os.listdir(experiments_dir):
    #     if file.startswith(experiment_name.lower()) and file.endswith(".txt"):
    #         filepath = os.path.join(experiments_dir, file)
    #         params = extract_params_from_filename(file)
    #         metrics = process_file(filepath)
    #         if metrics:
    #             mean_values, std_values = compute_mean_std2(metrics)
    #             result = {**params}
    #             for metric in mean_values:
    #                 result[f"{metric} Mean"] = mean_values[metric]
    #                 result[f"{metric} Std"] = std_values[metric]
    #             results.append(result)

    # # Convert results to DataFrame
    # df = pd.DataFrame(results)

    # # Generate visualizations
    # create_visualizations(df, plot_dir, experiment_name)

    # print(f"LaTeX table saved to {output_path}. Plots saved to {plot_dir}.")


    # for file in os.listdir(logs_dir):
    #     print(f"Processing file: {file}")
        
    #     if file.startswith(experiment_name.lower()) and file.endswith(".log"):
    #         # Extract parameters from filename
    #         filepath = os.path.join(logs_dir, file)
    #         match = re.search(r"b(\d+)_.*_ms(\d+)_mb(\d+)_.*", file)
    #         if not match:
    #             print(f"Filename {file} does not match expected format.")
    #             continue

    #     board_size, message_size, message_bits = map(int, match.groups())
    #     print(f"Processing: Board Size {board_size}, Message Size {message_size}, Message Bits {message_bits}")

    #     # Process file and plot results
    #     df = process_log_file(filepath)
    #     if not df.empty:
    #         plot_results(df, plot_dir, board_size, message_size, message_bits)
    #     else:
    #         print(f"No data found in {filepath}")

    plot_training_results(logs_dir, plot_dir, args.experiment_name)