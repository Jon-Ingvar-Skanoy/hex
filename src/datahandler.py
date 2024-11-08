# datahandler.py

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import re

class DataHandler:
    def __init__(self,
                 paths: dict = None,
                 files: dict = None,
                 dataset: str = '3x3_2000_40_0',
                 file_ext: str = '.csv',
                 dataloader: str = 'pd',
                 n_samples: int = 1000,
                 balance_data: bool = True,
                 perform_split: bool = True):
        
        self.paths = paths
        self.files = files
        self.dataset = dataset
        self.files_dict = {}
        self.file_ext = file_ext
        self.dataloader = dataloader
        self.n_samples = n_samples
        self.is_data_balanced = balance_data
        self.perform_split = perform_split

        self.data = None
        self.headers = None

        self.X = None
        self.y = None
        self.starting_player = None

        self.load_data_files(self.file_ext)
        self.load_data()
        self.prepare_data()

        if balance_data:
            self.X, self.y, self.starting_player = self.balance_data(self.X, self.y, self.starting_player, self.n_samples)
        
        if perform_split:
            self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()


    def load_data_files(self, file_ext=None):
        file_ext = file_ext or self.file_ext
        files = self.paths['data'].glob(f"*{file_ext}")
        
        for file in files:
            filename = file.stem
            # Regular expression to capture board_size, samples, op, and mbf
            match = re.match(r'(\d+x\d+)_(\d+)_(\d+)_(\d+)', filename)
            
            if match:
                board_size = match.group(1)  # e.g., "3x3" or "17x17"
                samples = int(match.group(2))  # e.g., "2000" or "200000"
                op = int(match.group(3))  # e.g., "40" or "60"
                mbf = int(match.group(4))  # e.g., "2" or "5"
                
                # Create a dictionary entry for this file
                self.files_dict[filename] = {
                    'board_size': board_size,
                    'samples': samples,
                    'op': op,
                    'mbf': mbf,
                    'file_path': file  # Store the actual path to the file
                }
            else:
                print(f"Filename {filename} does not match the expected pattern")
        
    def load_data(self, dataloader=None, n_samples=None):
        dataloader = dataloader or self.dataloader
        n_samples = n_samples or self.n_samples
        filename = self.files_dict.get(self.files.get('data'), None)
        if filename is None:
            raise ValueError(f"Dataset {self.files.get('data', 'UNKNOWN')} not found in files_dict.")
        else:
            filename = filename['file_path']
            
        if dataloader == 'pd':
            self.data = pd.read_csv(filename)
            self.headers = self.data.columns.tolist()
        elif dataloader == 'np.genfromtxt':
            with open(filename, 'r') as f:
                self.headers = f.readline().strip().split(',')

            self.data = np.genfromtxt(filename,
                                      delimiter=',',
                                      skip_header=1,
                                      dtype=np.int32,
                                      max_rows=n_samples*2)
        else:
            raise ValueError(f"Invalid dataloader: {dataloader}")
        self.dataset = self.files['data']
        
    def prepare_data(self):
        if self.dataloader == 'pd':
            # Expecting starting_player as the second-to-last column, winner as the last
            X = self.data.iloc[:, :-2]  # All columns except the last two (starting_player and winner)
            self.starting_player = self.data.iloc[:, -2].values  # Second-to-last column
            y = self.data.iloc[:, -1]  # Last column is winner
            self.X = X.values
            self.y = y.values
        elif self.dataloader == 'np.genfromtxt':
            # NumPy data format with "starting_player" as the second-to-last column
            X = self.data[:, :-2]
            self.starting_player = self.data[:, -2]
            y = self.data[:, -1]
            self.X = X
            self.y = y

    def balance_data(self, X=None, y=None, starting_player=None, n_samples=None):
        X = self.X if X is None else X
        y = self.y if y is None else y
        starting_player = self.starting_player if starting_player is None else starting_player
        n_samples = n_samples or self.n_samples
        
        # Identify unique class combinations based on starting_player and winner
        indices = [(starting_player == sp) & (y == wy) for sp in np.unique(starting_player) for wy in np.unique(y)]
        
        # Resample each class to ensure balanced starting player and outcome
        balanced_X, balanced_y, balanced_sp = [], [], []
        n_samples_per_class = n_samples // len(indices)
        for condition in indices:
            current_size = np.sum(condition)
            replace = current_size < n_samples_per_class
            X_resampled, y_resampled, sp_resampled = resample(
                X[condition], y[condition], starting_player[condition],
                replace=replace,
                n_samples=n_samples_per_class,
                random_state=42
            )
            balanced_X.append(X_resampled)
            balanced_y.append(y_resampled)
            balanced_sp.append(sp_resampled)
        
        self.X = np.vstack(balanced_X)
        self.y = np.hstack(balanced_y)
        self.starting_player = np.hstack(balanced_sp)
        
        return self.X, self.y, self.starting_player
    
    def split_data(self, X=None, y=None, test_size=0.2):
        X = self.X if X is None else X
        y = self.y if y is None else y
        return train_test_split(X, y, test_size=test_size, random_state=42)
    
    def show_data_info(self):
        print(f"Data shape: {self.data.shape}")
        print(f"X shape: {self.X.shape}")
        print(f"y shape: {self.y.shape}")
        print(f"Starting player shape: {self.starting_player.shape}")
        print(f"Headers: {self.headers}")
        if self.perform_split:
            print(f"X_train shape: {self.X_train.shape}")
            print(f"y_train shape: {self.y_train.shape}")
            print(f"X_test shape: {self.X_test.shape}")
            print(f"y_test shape: {self.y_test.shape}")

    def get_class_distribution(self):
        unique, counts = np.unique(self.y, axis=0, return_counts=True)
        return dict(zip([f"{sp}-{w}" for sp, w in unique], counts))
    
    def count_unique_games(self):
        """
        Count and return the number of unique games in the dataset.
        Assumes that each row in self.X represents a single game.
        """
        if self.X is None:
            raise ValueError("Data has not been loaded or prepared.")
        
        unique_games = np.unique(self.X, axis=0)  # Find unique rows in X
        num_unique_games = unique_games.shape[0]  # Count unique rows
        
        return num_unique_games

    def save_graphs(self,
                    graphs_train,
                    graphs_test,
                    X_train = None,
                    y_train = None,
                    X_test = None,
                    y_test = None,
                    output_file = None):

        X_train = X_train or self.X_train
        y_train = y_train or self.y_train
        X_test = X_test or self.X_test
        y_test = y_test or self.y_test
        if output_file:
            output_file = f"{self.paths['graphs']}/{output_file}"
        else:
            output_file = f"{self.paths['graphs']}/graphs_{self.dataset}_{self.n_samples}.pkl"

        with open(output_file, 'wb') as f:
            pickle.dump((graphs_train, graphs_test, X_train, y_train, X_test, y_test), f)
    
    def load_graphs(self, input_file = None):
        if input_file:
            input_file = f"{self.paths['graphs']}/{input_file}"
        else:
            input_file = f"{self.paths['graphs']}/graphs_{self.dataset}_{self.n_samples}.pkl"
            
        with open(input_file, 'rb') as f:
            graphs_train, graphs_test, X_train, y_train, X_test, y_test = pickle.load(f)
        return graphs_train, graphs_test, X_train, y_train, X_test, y_test
