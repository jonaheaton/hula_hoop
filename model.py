import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Optional


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
    def __init__(self, input_size: int=1,
                 hidden_size: int=32,
                 num_layers: int=1,
                 output_size: int=1,
                 data_scaler: Optional[MinMaxScaler] = None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.net = LSTMModule(input_size, hidden_size, num_layers, output_size)
        self.optimizer = self._get_optimizer()
        self.criterion = self._get_criterion()
        self.data_scaler = data_scaler

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

    def _reshape_and_scale_data(self, x: np.ndarray) -> np.ndarray:
        if self.data_scaler is not None:
            if len(x.shape) > 2:
                x_temp = x.reshape(-1, self.data_scaler.n_features_in_)
                x_temp = self.data_scaler.transform(x_temp)
                x = x_temp.reshape(-1, x.shape[1], x.shape[2])
            else:
                x = self.data_scaler.transform(x)
        return x

    def fit(self, x: np.ndarray, y: np.ndarray, 
            num_epoch: int = 100, 
            batch_size: int = 128):

        if self.data_scaler is not None:
            self.data_scaler.fit(y)
        y = self._reshape_and_scale_data(y)
        x = self._reshape_and_scale_data(x)

        dataset = torch.utils.data.TensorDataset(torch.Tensor(x), torch.Tensor(y))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for current_epoch in range(num_epoch):
            loss = self._train_one_epoch(dataloader)
            if current_epoch % 10 == 0:
                print(f'Epoch {current_epoch}:\t loss={loss:.4f}')

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = self._reshape_and_scale_data(x)
        self.net.eval()
        with torch.inference_mode():
            logits= self.net(torch.Tensor(x))
        y = logits.clone().detach().cpu().numpy()
        if self.data_scaler is not None:
            y = self.data_scaler.inverse_transform(y)
        return y
    
    def generate(self, x:np.ndarray, num_steps:int) -> np.ndarray:
        x = self._reshape_and_scale_data(x)
        self.net.eval()
        x = torch.Tensor(x, dtype=torch.float32)
        y_generated = []
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        with torch.inference_mode():
            for i in range(num_steps):
                y_pred = self.net(x)
                y_generated.append(y_pred.clone().detach().cpu().numpy())
                x = torch.cat([x[:, 1:, :], y_pred.unsqueeze(0)], dim=1)
        # y = x.clone().detach().cpu().numpy()
        y = np.array(y_generated)
        y = y.reshape(-1, y.shape[-1])
        if self.data_scaler is not None:
            y = self.data_scaler.inverse_transform(y)
        return y

    def evaluate(self, x: np.ndarray, y: np.ndarray, col_names: Optional[list]= None) -> dict:
        self.net.eval()
        # y_pred = self.net(torch.Tensor(x))
        y_pred = self.predict(x)
        loss_dict = {}
        if col_names is not None:
            assert len(col_names) == y_pred.shape[1]
            y_pred = pd.DataFrame(y_pred, columns=col_names)
            y = pd.DataFrame(y, columns=col_names)
            
            if self.data_scaler is not None:
                y_pred_sc = self.data_scaler.transform(y_pred.values)
                y_sc = self.data_scaler.transform(y.values)
                y_pred_sc = pd.DataFrame(y_pred_sc, columns=col_names)
                y_sc = pd.DataFrame(y_sc, columns=col_names)

            for col in col_names:
                loss = self.criterion(torch.Tensor(y_pred[col]), torch.Tensor(y[col]))
                loss_dict[col] = loss.item()

                if self.data_scaler is not None:
                    loss = self.criterion(torch.Tensor(y_pred_sc[col]), torch.Tensor(y_sc[col]))
                    loss_dict[col + '_scaled'] = loss.item()

        if self.data_scaler is not None:
            y_pred_sc = self.data_scaler.transform(y_pred)
            y_sc = self.data_scaler.transform(y)
            loss = self.criterion(torch.Tensor(y_pred_sc), torch.Tensor(y_sc))
        else:
            loss = self.criterion(torch.Tensor(y_pred), torch.Tensor(y))
        loss_dict['total'] = loss.item()

        return loss_dict  