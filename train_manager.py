from data_manager import Data_Manager
from torch import nn
import torch
from tqdm.auto import tqdm

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
        total_loss, total_acc = 0, 0

        self.model.train()

        for X, y in self.data_manager.train_dataloader:
            y_logits = self.model(X)

            loss = self.loss_fn(y_logits, y)
            total_loss += loss.item()

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            labels = torch.round(torch.sigmoid(y_logits))
            total_acc += (labels==y).sum().item() / len(y)

        total_loss /= len(self.data_manager.train_dataloader)
        total_acc /= len(self.data_manager.train_dataloader)

        return total_loss, total_acc
    
    def test_loop(self):
        total_loss, total_acc = 0, 0

        self.model.eval()
        with torch.inference_mode():
            for X, y in self.data_manager.test_dataloader:
                y_logits = self.model(X)

                loss = self.loss_fn(y_logits, y)
                total_loss += loss.item()

                labels = torch.round(torch.sigmoid(y_logits))
                total_acc += (labels==y).sum().item() / len(y)

        total_loss /= len(self.data_manager.test_dataloader)
        total_acc /= len(self.data_manager.test_dataloader)

        return total_loss, total_acc
    
    def train_model(self,
                    epochs: int,
                    printInterval: int):
        for epoch in tqdm(range(epochs), desc="Training model"):
            train_loss, train_acc = self.train_loop()

            test_loss, test_acc = self.test_loop()

            if epoch == epochs - 1 or epoch % printInterval == 0:
                print(f"Epoch: {epoch} | Train loss: {train_loss:.4f} Train acc: {train_acc:.4f}% | Test loss {test_loss:.4f} Test acc: {test_acc:.4f}%")
