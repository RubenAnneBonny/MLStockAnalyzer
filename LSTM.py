from torch import nn
from data_manager import Data_Manager
from train_manager import Train_Manager

class LSTMModel(nn.Module):
    def __init__(self, 
                 hidden_size: int,
                 num_layers: int):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)

        out = out[:, -1, :]

        out = self.linear(out)

        return out
    
def main():
    model = LSTMModel(hidden_size=20,
                      num_layers=3)

    data_manager = Data_Manager(train_split=0.8,
                                shuffle_data_before_split=True,
                                stock="AAPL",
                                target_binary=False,
                                data_binary=False,
                                window_size=20,
                                batch_size=32)
    
    train_manager = Train_Manager(data_manager=data_manager,
                                  model=model)
    
    train_manager.train_model(epochs=80,
                              print_interval=8)
    
    train_manager.evaluate_model()
    
main()