import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Optional


class LSTMModule(torch.nn.Module):
    """
    A PyTorch module that implements a Long Short-Term Memory (LSTM) model.

    Args:
        input_size (int): The number of expected features in the input x.
        hidden_size (int, optional): The number of features in the hidden state h. Default is 32.
        num_layers (int, optional): Number of recurrent layers. Default is 1.
        output_size (int, optional): The number of output features. Default is 1.

    Attributes:
        input_size (int): The number of expected features in the input x.
        hidden_size (int): The number of features in the hidden state h.
        num_layers (int): Number of recurrent layers.
        output_size (int): The number of output features.
        lstm (torch.nn.LSTM): The LSTM layer.
        linear (torch.nn.Linear): The linear layer.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Performs a forward pass through the LSTM module.

    """

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
        """
        Performs a forward pass through the LSTM module.

        Args:
            x (torch.Tensor): The input tensor of shape (batch, sequence, feature).

        Returns:
            torch.Tensor: The output tensor of shape (batch, output_size).

        """
        # Forward pass through LSTM layer
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Assuming we only care about the last output for a sequence-to-one model
        # Selecting the last output for each sequence
        last_out = lstm_out[:, -1, :]
        
        # Forward pass through Linear layer using the last output of the LSTM
        out = self.linear(last_out)
        return out


class HulaHoopLSTM:
    """
    A class representing a Long Short-Term Memory (LSTM) model for Hula Hoop prediction.

    Parameters:
    - input_size (int): The number of expected features in the input x.
    - hidden_size (int): The number of features in the hidden state h.
    - num_layers (int): Number of recurrent layers. Default is 1.
    - output_size (int): The number of output features. Default is 1.
    - data_scaler (Optional[MinMaxScaler]): An optional data scaler to normalize the input data.

    Attributes:
    - input_size (int): The number of expected features in the input x.
    - hidden_size (int): The number of features in the hidden state h.
    - num_layers (int): Number of recurrent layers.
    - output_size (int): The number of output features.
    - net (LSTMModule): The LSTM module used for prediction.
    - optimizer (torch.optim.Optimizer): The optimizer used for training.
    - criterion (torch.nn.Module): The loss function used for training.
    - data_scaler (Optional[MinMaxScaler]): The data scaler used for normalization.

    Methods:
    - _get_optimizer(): Setup a PyTorch optimizer.
    - _get_criterion(): Setup a loss function.
    - _train_one_epoch(dataloader: torch.utils.data.DataLoader) -> torch.Tensor:
        Train one epoch with a data loader and return the average train loss.
    - _reshape_and_scale_data(x: np.ndarray) -> np.ndarray:
        Reshape and scale the input data using the data scaler.
    - fit(x: np.ndarray, y: np.ndarray, num_epoch: int = 100, batch_size: int = 128):
        Train the model on the input data.
    - predict(x: np.ndarray) -> np.ndarray:
        Make predictions on the input data.
    - generate(x: np.ndarray, num_steps: int) -> np.ndarray:
        Generate new data points based on the input data.
    - evaluate(x: np.ndarray, y: np.ndarray, col_names: Optional[list] = None) -> dict:
        Evaluate the model's performance on the input data.

    """

    def __init__(self, input_size: int=1, hidden_size: int=32, num_layers: int=1,
                 output_size: int=1, data_scaler: Optional[MinMaxScaler] = None):
        """
        Initialize the HulaHoopLSTM model.

        Parameters:
        - input_size (int): The number of expected features in the input x.
        - hidden_size (int): The number of features in the hidden state h.
        - num_layers (int): Number of recurrent layers. Default is 1.
        - output_size (int): The number of output features. Default is 1.
        - data_scaler (Optional[MinMaxScaler]): An optional data scaler to normalize the input data.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.net = LSTMModule(input_size, hidden_size, num_layers, output_size)
        self.optimizer = self._get_optimizer()
        self.criterion = self._get_criterion()
        self.data_scaler = data_scaler

    def _get_optimizer(self) -> torch.optim.Optimizer:
        """
        Setup a PyTorch optimizer.

        Returns:
        - torch.optim.Optimizer: The optimizer for training the model.
        """
        return torch.optim.AdamW(self.net.parameters(), lr=1e-3, weight_decay=1e-5)

    def _get_criterion(self) -> torch.nn.Module:
        """
        Setup a loss function.

        Returns:
        - torch.nn.Module: The loss function for training the model.
        """
        return nn.MSELoss()

    def _train_one_epoch(self, dataloader: torch.utils.data.DataLoader) -> torch.Tensor:
        """
        Train one epoch with data loader and return the average train loss.

        Parameters:
        - dataloader (torch.utils.data.DataLoader): The data loader for training.

        Returns:
        - torch.Tensor: The average train loss for the epoch.
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
        """
        Reshape and scale the input data using the data scaler.

        Parameters:
        - x (np.ndarray): The input data to be reshaped and scaled.

        Returns:
        - np.ndarray: The reshaped and scaled input data.
        """
        if self.data_scaler is not None:
            if len(x.shape) > 2:
                x_temp = x.reshape(-1, self.data_scaler.n_features_in_)
                x_temp = self.data_scaler.transform(x_temp)
                x = x_temp.reshape(-1, x.shape[1], x.shape[2])
            else:
                x = self.data_scaler.transform(x)
        return x

    def fit(self, x: np.ndarray, y: np.ndarray, num_epoch: int = 100, batch_size: int = 128):
        """
        Train the model on the input data.

        Parameters:
        - x (np.ndarray): The input data for training.
        - y (np.ndarray): The target data for training.
        - num_epoch (int): The number of training epochs. Default is 100.
        - batch_size (int): The batch size for training. Default is 128.
        """
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
        """
        Make predictions on the input data.

        Parameters:
        - x (np.ndarray): The input data for prediction.

        Returns:
        - np.ndarray: The predicted output data.
        """
        x = self._reshape_and_scale_data(x)
        self.net.eval()
        with torch.inference_mode():
            logits = self.net(torch.Tensor(x))
        y = logits.clone().detach().cpu().numpy()
        if self.data_scaler is not None:
            y = self.data_scaler.inverse_transform(y)
        return y

    def generate(self, x: np.ndarray, num_steps: int) -> np.ndarray:
        """
        Generate new data points based on the input data.

        Parameters:
        - x (np.ndarray): The input data for generating new data points.
        - num_steps (int): The number of steps to generate.

        Returns:
        - np.ndarray: The generated data points.
        """
        x = self._reshape_and_scale_data(x)
        self.net.eval()
        x = torch.tensor(x, dtype=torch.float32)
        y_generated = []
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        with torch.inference_mode():
            for i in range(num_steps):
                y_pred = self.net(x)
                y_generated.append(y_pred.clone().detach().cpu().numpy())
                x = torch.cat([x[:, 1:, :], y_pred.unsqueeze(0)], dim=1)
        y = np.array(y_generated)
        y = y.reshape(-1, y.shape[-1])
        if self.data_scaler is not None:
            y = self.data_scaler.inverse_transform(y)
        return y

    def evaluate(self, x: np.ndarray, y: np.ndarray, col_names: Optional[list] = None) -> dict:
        """
        Evaluate the model's performance on the input data.

        Parameters:
        - x (np.ndarray): The input data for evaluation.
        - y (np.ndarray): The target data for evaluation.
        - col_names (Optional[list]): The column names of the target data. Default is None.

        Returns:
        - dict: A dictionary containing the loss values for each column and the total loss.
        """
        self.net.eval()
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
