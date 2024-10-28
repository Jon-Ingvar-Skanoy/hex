# dbhandler.py

import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import optuna

class DBHandler:
    def __init__(self):
        # Load environment variables for database credentials
        load_dotenv()
        self.db_host = os.getenv("DB_HOST")
        self.db_port = os.getenv("DB_PORT")
        self.db_user = os.getenv("DB_USER")
        self.db_password = os.getenv("DB_PASSWORD")
        self.db_name = os.getenv("DB_NAME")
        
        self.engine = create_engine(f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}")
        self.target_storage_url = f"postgresql+psycopg2://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
    
    def execute_query(self, query):
        """Executes a query and returns a DataFrame."""
        with self.engine.connect() as connection:
            return pd.read_sql_query(query, connection)

    def load_study(self, study_name, storage_url):
        """Loads a study from a specified storage URL."""
        return optuna.load_study(study_name=study_name, storage=storage_url)

    def get_study_names(self):
        """Fetches all study names from the database."""
        query = "SELECT study_id, study_name FROM studies"
        studies_df = self.execute_query(query)
        return {row['study_name']: row['study_id'] for _, row in studies_df.iterrows()}

    def study_exists(self, study_name):
        """Checks if a study exists in the target storage."""
        try:
            self.load_study(study_name, self.target_storage_url)
            return True
        except KeyError:
            return False

    def delete_study(self, study_name):
        """Deletes a study if it exists in the target storage."""
        if self.study_exists(study_name):
            confirmation = input(f"The study '{study_name}' already exists in the target database. Do you want to delete it? (yes/no): ").strip().lower()
            if confirmation == 'yes':
                try:
                    optuna.delete_study(study_name=study_name, storage=self.target_storage_url)
                    print("Study deleted successfully.")
                except Exception as e:
                    print(f"Error deleting study: {e}")
            else:
                print("Deletion canceled.")
                return False
        return True

    def copy_study(self, study_name, source_storage_url):
        """Copies a study from source storage to target storage."""
        try:
            optuna.copy_study(
                from_study_name=study_name,
                from_storage=source_storage_url,
                to_storage=self.target_storage_url,
                to_study_name=study_name
            )
            print("Study copied successfully.")
        except Exception as e:
            print(f"Error copying study: {e}")

    def get_trial_count(self, study_name):
        """Loads a study and returns the count of trials in it."""
        try:
            target_study = self.load_study(study_name, self.target_storage_url)
            return len(target_study.trials)
        except Exception as e:
            print(f"Error loading study: {e}")
            return None

    def load_board_sizes(self, study_id):
        """Fetches available board sizes for a given study ID."""
        query = f"""
            SELECT DISTINCT param_value::int 
            FROM trial_params 
            WHERE param_name = 'board_size' 
              AND trial_id IN (SELECT trial_id FROM trials WHERE study_id={study_id})
        """
        board_sizes_df = self.execute_query(query)
        return board_sizes_df['param_value'].sort_values().tolist()

    def query_top_results(self, study_id, board_size):
        """Fetches top results with all parameters and user attributes."""
        query = f"""
            SELECT t.trial_id AS Number, t.state AS State, v.value AS Value, 
                   p.param_name AS paramname, p.param_value AS paramvalue, 
                   u.key AS userattributekey, u.value_json AS userattributevalue
            FROM trials t
            JOIN trial_params p ON t.trial_id = p.trial_id
            LEFT JOIN trial_user_attributes u ON t.trial_id = u.trial_id
            LEFT JOIN trial_values v ON t.trial_id = v.trial_id
            WHERE t.study_id = {study_id} AND t.state = 'COMPLETE'
                AND t.trial_id IN (
                    SELECT trial_id FROM trial_params 
                    WHERE param_name = 'board_size' AND param_value::int = {board_size}
                )
        """
        return self.execute_query(query)

    def close_connection(self):
        """Closes the database engine connection pool."""
        self.engine.dispose()
