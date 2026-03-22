import yfinance as yf
import pandas as pd
import numpy as np
import time
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class Data_Manager:
    def __init__(self, 
                train_split: float = 0.8,
                shuffle_data_before_split: bool = False,
                stock: str = "AAPL",
                target_binary: bool = False,
                data_binary: bool = False,
                window_size: int = 10, 
                batch_size: int = 32,
                normalize_data: bool = True):

        self.today = self.get_todays_date()
        print(f"Collecting all data of {stock} until {self.today}")
        self.data = yf.download(stock, start="1700-01-01", end=self.today)
        self.target_binary = target_binary
        self.data_binary = data_binary
        self.train_split = train_split
        self.window_size = window_size
        self.shuffle_data_before_split = shuffle_data_before_split
        self.batch_size = batch_size
        self.normalize_data = normalize_data

        self.X, self.y = self.create_input_data_windows()
        self.X_train, self.X_test, self.y_train, self.y_test = self.convert_data_to_tensor()

        self.train_dataloader = self.create_dataloader(self.X_train, self.y_train, True)
        self.test_dataloader = self.create_dataloader(self.X_test, self.y_test, False)

    def get_todays_date(self) -> str:
        lt = time.localtime()
        year = str(lt.tm_year)
        month = str(lt.tm_mon)
        if len(month) == 1:
            month = "0" + month
        day = str(lt.tm_mday)
        if len(day) == 1:
            day = "0" + day

        return year + "-" + month + "-" + day

    def convert_data_to_tensor(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, 
            train_size=self.train_split,
            shuffle=self.shuffle_data_before_split
        )

        X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
        X_test  = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
        y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
        y_test  = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

        return X_train, X_test, y_train, y_test
    
    def create_dataloader(self, X, y, shuffle):
        dataset = SequenceDataset(X, y)

        dataloader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                shuffle = shuffle)
        
        return dataloader

    def convert_num_to_date(self,
                            num: int) -> str:
        result = str(self.data.axes[0][num]).split()[0]

        return result
        
    def get_value(self,
                  index: int) -> np.array:
        return self.data.loc[self.convert_num_to_date(index)]["Close"].item()
    
    def convert_data_to_percent(self):
        numbers = []

        for i in tqdm(range(len(self.data) - 1), desc="Converting data to percentage"):
            numbers.append(1 - self.get_value(i + 1) / self.get_value(i))

        if self.normalize_data:
            mean, std = np.mean(numbers), np.std(numbers)
            numbers = (numbers - mean) / std

        return numbers
    
    def create_input_data_windows(self):
        percentages = self.convert_data_to_percent()

        data = []
        targets = []
        for i in range(len(percentages) - (self.window_size - 1) - 2):
            data.append([])
        
        for i in tqdm(range(len(data)), desc=f"Collecting data in windows of size {self.window_size}"):
            for index in range(i, i + self.window_size):
                if self.data_binary:
                    data[i].append(0 if percentages[index] < 0 else 1)
                else:
                    data[i].append(percentages[index])
            
            if self.target_binary:
                targets.append(0 if percentages[i + self.window_size + 1] < 0 else 1)
            else:
                targets.append(percentages[i + self.window_size + 1])

        return data, targets