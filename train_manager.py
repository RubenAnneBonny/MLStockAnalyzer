from data_manager import Data_Manager
from torch import nn
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np

class Train_Manager:
    def __init__(self,
                 data_manager: Data_Manager,
                 model: nn.Module):
        self.data_manager = data_manager
        self.model = model
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                          lr=0.001)
        self.loss_fn = nn.MSELoss()
        if data_manager.target_binary:
            self.loss_fn = nn.BCEWithLogitsLoss()

    def train_loop(self):
        total_loss = 0

        self.model.train()

        for X, y in self.data_manager.train_dataloader:
            y_logits = self.model(X)

            loss = self.loss_fn(y_logits, y)
            total_loss += loss.item()

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

        total_loss /= len(self.data_manager.train_dataloader)

        return total_loss
    
    def test_loop(self):
        total_loss = 0

        self.model.eval()
        with torch.inference_mode():
            for X, y in self.data_manager.test_dataloader:
                y_logits = self.model(X)

                loss = self.loss_fn(y_logits, y)
                total_loss += loss.item()

        total_loss /= len(self.data_manager.test_dataloader)

        return total_loss
    
    def train_model(self,
                    epochs: int,
                    print_interval: int):
        for epoch in range(epochs):
            train_loss = self.train_loop()

            test_loss = self.test_loop()

            if epoch == epochs - 1 or epoch % print_interval == 0:
                print(f"Epoch: {epoch} | Train loss: {train_loss:.6f} | Test loss {test_loss:.6f}")

    def evaluate_model(self):
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.inference_mode():
            for X, y in self.data_manager.test_dataloader:
                y_pred = self.model(X)
                all_preds.append(y_pred.detach().cpu())
                all_targets.append(y.detach().cpu())

        # Concatenate all batches
        all_preds = torch.cat(all_preds).numpy().flatten()
        all_targets = torch.cat(all_targets).numpy().flatten()

        # Denormalize if needed
        if self.data_manager.normalize_data:
            mean = np.mean(self.data_manager.convert_data_to_percent())
            std = np.std(self.data_manager.convert_data_to_percent())
            all_preds = all_preds * std + mean
            all_targets = all_targets * std + mean

        # Scatter plot predictions vs true values
        plt.figure(figsize=(8,8))
        plt.scatter(all_targets, all_preds, alpha=0.5)
        plt.plot([all_targets.min(), all_targets.max()],
                [all_targets.min(), all_targets.max()],
                'r--', label='Perfect prediction')
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.title("Predictions vs True Values")
        plt.legend()
        plt.show()
