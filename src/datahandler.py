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
                 dataset: str = '3x3_0',
                 file_ext: str = '.csv',
                 dataloader: str = 'pd',
                 n_samples: int = 100000,
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

        self.load_data_files(self.file_ext)
        self.load_data()
        self.prepare_data()

        if balance_data:
            self.X, self.y = self.balance_data(self.X, self.y, self.n_samples)
        
        if perform_split:
            self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()


    def load_data_files(self, file_ext=None):
        file_ext = file_ext or self.file_ext
        files = self.paths['data'].glob(f"*{file_ext}")
        
        for file in files:
            filename = file.stem

            base_name = filename.split('_')[0]

            match = re.search(r'_([0-9])$', filename)
            
            if match:
                numeric_suffix = match.group(1)
                short_name = f"{base_name}_{numeric_suffix}"
            else:
                short_name = f"{filename}_0"
            
            self.files_dict[short_name] = file
        
    def load_data(self, dataloader=None, n_samples=None):
        dataloader = dataloader or self.dataloader
        n_samples = n_samples or self.n_samples
        filename = self.files_dict[self.files['data']]
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
            X = self.data.iloc[:, :-1]
            y = self.data.iloc[:, -1]
            self.X = X.values
            self.y = y.values
        elif self.dataloader == 'np.genfromtxt':
            X = self.data[:, :-1]
            y = self.data[:, -1]
            self.X = X
            self.y = y

    def balance_data(self, X=None, y=None, n_samples=None):
        X = self.X if X is None else X
        y = self.y if y is None else y
        n_samples = n_samples or self.n_samples
        class_counts = np.bincount(y)
        majority_class = np.argmax(class_counts)
        minority_class = np.argmin(class_counts)

        X_majority = X[y == majority_class]
        X_minority = X[y == minority_class]

        n_samples_per_class = n_samples // 2

        X_majority_downsampled, y_majority_downsampled = resample(
            X_majority, y[y == majority_class],
            replace=False,  
            n_samples=n_samples_per_class, 
            random_state=42 
        )

        X_minority_downsampled, y_minority_downsampled = resample(
            X_minority, y[y == minority_class],
            replace=False,  
            n_samples=n_samples_per_class, 
            random_state=42  
        )

        X_resampled = np.vstack((X_majority_downsampled, X_minority_downsampled))
        y_resampled = np.hstack((y_majority_downsampled, y_minority_downsampled))

        return X_resampled, y_resampled
    
    def split_data(self, X=None, y=None, test_size=0.2):
        X = self.X if X is None else X
        y = self.y if y is None else y
        return train_test_split(X, y, test_size=test_size, random_state=42)
    
    def show_data_info(self):
        print(f"Data shape: {self.data.shape}")
        print(f"X shape: {self.X.shape}")
        print(f"y shape: {self.y.shape}")
        print(f"Headers: {self.headers}")
        if self.perform_split:
            print(f"X_train shape: {self.X_train.shape}")
            print(f"y_train shape: {self.y_train.shape}")
            print(f"X_test shape: {self.X_test.shape}")
            print(f"y_test shape: {self.y_test.shape}")

    def get_class_distribution(self):
        return np.bincount(self.y)

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
