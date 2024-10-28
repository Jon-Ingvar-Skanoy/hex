# run_optuna_dashboard.py

import argparse
from optuna_dashboard import run_server
from src.dbhandler import DBHandler

parser = argparse.ArgumentParser(description="Run Optuna dashboard server")
parser.add_argument(
    "--port",
    type=int,
    default=8080,
    help="Port number to run the Optuna dashboard on (default: 8080)"
)
args = parser.parse_args()

# Set up the database connection
db = DBHandler()
db_name = db.db_name
postgresql_url = db.target_storage_url

if __name__ == "__main__":
    print(f"Starting Optuna dashboard connected to database: {db_name}")
    run_server(postgresql_url, port=args.port)
