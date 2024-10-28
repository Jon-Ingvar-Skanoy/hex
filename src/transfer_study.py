# transfer_study.py
# This is VERY SLOW when the number of trials increase (100<).

import os
from src.dbhandler import DBHandler

# Define paths and source study URL
path_script = os.path.dirname(os.path.abspath(__file__))
path_project = os.path.dirname(path_script)
path_results = os.path.join(path_project, 'results')
path_optuna = os.path.join(path_results, 'optuna')
source_storage_url = f"sqlite:///{path_optuna}/ja_tsehex.db"
study_name = "Study_4x4_0"

# Instantiate the database handler
db_handler = DBHandler()

# Check if study exists and delete if confirmed
if db_handler.study_exists(study_name):
    if not db_handler.delete_study(study_name):
        print("Exiting due to study deletion cancellation.")
        exit()

# Copy the study to the target storage
db_handler.copy_study(study_name, source_storage_url)

# Load and verify the copied study
trial_count = db_handler.get_trial_count(study_name)
if trial_count is not None:
    print(f"Number of trials in the copied target study: {trial_count}")
    print("Copy complete.")
else:
    print("Error during verification of copied study.")