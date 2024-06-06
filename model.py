import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable 


class LSTMModule(torch.nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1, output_size=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.lstm = torch.nn.LSTM(input_size=input_size, 
                                  hidden_size=hidden_size, 
                                  num_layers=num_layers, 
                                  batch_first=True)
        self.linear = torch.nn.Linear(in_features=hidden_size, 
                                      out_features=output_size)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through LSTM layer
        # x should be of shape (batch, sequence, feature)
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Assuming we only care about the last output for a sequence-to-one model
        # Selecting the last output for each sequence
        last_out = lstm_out[:, -1, :]
        
        # Forward pass through Linear layer using the last output of the LSTM
        out = self.linear(last_out)
        return out


class HulaHoopLSTM:
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, output_size=1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.net = LSTMModule(input_size, hidden_size, num_layers, output_size)
        self.optimizer = self._get_optimizer()
        self.criterion = self._get_criterion()

    def _get_optimizer(self) -> torch.optim.Optimizer:
        """Setup a PyTorch optimizer."""
        # some weight decay combats overfitting
        return torch.optim.AdamW(self.net.parameters(), lr=1e-3, weight_decay=1e-5)

    def _get_criterion(self) -> torch.nn.Module:
        """Setup a loss function."""
        return nn.MSELoss()

    def _train_one_epoch(self, dataloader: torch.utils.data.DataLoader) -> torch.Tensor:
        """Train one epoch with data loader and return the average train loss.

        Please make use of the following initialized items:
        - self.net
        - self.optimizer
        - self.criterion

        """
        self.net.train()
        total_loss = 0
        for batch in dataloader:
            x, y = batch
            self.optimizer.zero_grad()
            y_pred = self.net(x)
            loss = self.criterion(y_pred, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def fit(self, x: np.ndarray, y: np.ndarray, num_epoch: int = 100, batch_size: int = 128):
        dataset = torch.utils.data.TensorDataset(torch.Tensor(x), torch.Tensor(y))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for current_epoch in range(num_epoch):
            loss = self._train_one_epoch(dataloader)
            if current_epoch % 10 == 0:
                print(f'Epoch {current_epoch}:\t loss={loss:.4f}')

    def predict(self, x: np.ndarray):
        self.net.eval()
        logits= self.net(torch.Tensor(x))
        return logits.detach().cpu().numpy()