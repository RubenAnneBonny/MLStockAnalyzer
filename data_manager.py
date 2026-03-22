import yfinance as yf
import pandas as pd
import numpy as np
import time
from tqdm.auto import tqdm

class Data_Manager:
    def __init__(self, 
                train_split: float = 0.8,
                stock: str = "AAPL",
                target_binary: bool = False,
                data_binary: bool = False,
                window_size: int = 10):

        self.today = self.get_todays_date()
        print(f"Collecting all data of {stock}  until {self.today}")
        self.data = yf.download(stock, start="1700-01-01", end=self.today)
        self.target_binary = target_binary
        self.data_binary = data_binary
        self.train_split = train_split
        self.window_size = window_size

        self.X, self.y = self.create_input_data_windows()

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
    
manager = Data_Manager(target_binary=True)

print(manager.y[:10])