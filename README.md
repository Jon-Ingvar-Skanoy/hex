# TseHex

### Applying Graph Tsetlin Machine to Hex Game Predictions

## Overview 
This repository contains the implementation of the research paper *"TseHex"*. The Graph Tsetlin Machine (GTM) is an extension of the Tsetlin Machine designed to handle graph-structured data. This project demonstrates the application of GTM for predicting outcomes in the game of Hex, an abstract, combinatorial two-player board game with no ties and high strategic complexity.
The study includes:

- Dataset generation for Hex game states.

- Graph construction methods and encoding schemes for representing Hex boards.

- Hyperparameter tuning and performance analysis of GTM across varying board sizes.

- Exploration of interpretability through logical clause analysis.

## Features 
 
- **Dataset generation** : Generate datasets of Hex game states using random play.
 
- **Graph construction** : Encode Hex game states as directed graphs with features.
 
- **GTM implementation** : Use GTM to predict game outcomes.
 
- **Hyperparameter tuning** : Optimize GTM performance using Optuna.
 
- **Scalability** : Analyze GTM performance across board sizes (from 4x4 to 15x15).
 
- **Interpretability** : Understand GTM's decision-making via logical clause analysis.

## Repository Structure 
`src` Folder
Contains the main scripts and notebooks used in the project:
 
- `compare_datasets.ipynb`: Compare datasets generated under different conditions.
 
- `experiment*.py`: Scripts for running experiments with GTM.
 
- `graph_gen*.ipynb`: Scripts for processing datasets and generating graphs.
 
- `optuna_global.ipynb`: Hyperparameter tuning using Optuna.
 
- `parallel_runner.py`: Utility for running experiments in parallel.
 
- `print_clauses.py`: Print and analyze GTM clauses.
 
- `read_clauses.py`: Read GTM clause data.
 
- `requirements.txt`: Python dependencies.


`src/` Contains utility modules for data and graph handling:
 
- `datahandler.py`: Functions for loading, balancing, and splitting datasets.
 
- `dbhandler.py`: Database handling for Optuna studies.
 
- `graphhandler.py`: Helper functions for graph operations.
 
- `helper_functions.py`: Miscellaneous utility functions.
 
- `run_optuna_dashboard.py`: Script for running the Optuna dashboard.
 
- `transfer_study.py`: Scripts for migrating Optuna studies.

## Getting Started 

### Prerequisites 
- [GraphTsetlinMachine](https://github.com/cair/GraphTsetlinMachine)

- Python 3.8+
 
- Dependencies listed in `requirements.txt`.

### Installation 
 
1. Clone the repository:

```bash
git clone https://github.com/Jon-Ingvar-Skanoy/hex
cd hex
```
 
2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Usage 

#### 1. Generate Datasets 
Use `main.cpp` (https://github.com/Jon-Bull/simhex) to generate Hex game datasets with different configurations. Save datasets in the format described in the paper.
#### 2. Preprocess Data 
Run `graph_gen*.ipynb` notebooks to preprocess datasets and generate graph objects.
#### 3. Train GTM 
Use scripts like `experiment.py` or `experiment_hypervector.py` to train the GTM on processed datasets.
#### 4. Tune Hyperparameters 
Run `optuna_global.ipynb` for hyperparameter tuning using AWS or local storage.
#### 5. Analyze Results 
Utilize `print_clauses.py` and related scripts to interpret GTM's decision-making.
## Results 
 
- Achieved accuracy of **99.46%**  on 11x11 boards in final states.

- Accuracy on larger boards declines, highlighting scalability challenges.

- Demonstrated interpretability through logical clause analysis.

## Future Work 

- Optimize graph representations for better scalability.

- Explore self-play mechanisms for enhanced dataset generation.




#### Other

"This README was mostly generated by ChatGPT-4o"

For more details, refer to the full paper.