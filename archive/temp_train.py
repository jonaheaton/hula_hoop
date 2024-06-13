import pandas as pd
import numpy as np
from model import HulaHoopLSTM
import torch




# def create_dataset(dataset, lookback):
#     """Transform a time series into a prediction dataset
    
#     Args:
#         dataset: A numpy array of time series, first dimension is the time steps
#         lookback: Size of window for prediction
#     """
#     X, y = [], []
#     for i in range(len(dataset)-lookback):
#         feature = dataset[i:i+lookback,:]
#         target = dataset[i+1:i+lookback+1,:]
#         X.append(feature)
#         y.append(target)
#     return torch.tensor(X), torch.tensor(y)

# Load the data
input_data_file = '/Users/jonaheaton/Documents/hulahoop_data/output_DSC_7450_hula_hoop_data.csv'
df = pd.read_csv(input_data_file)
df =df[['top ellipse X','top ellipse Y','bottom ellipse X','bottom ellipse Y']].copy()

# look for outliers and replace them with the average of the previous and next values
for col in df.columns:
    df[col] = df[col].mask(df[col].diff().abs() > 100, np.nan)
    df[col] = df[col].fillna((df[col].shift(1) + df[col].shift(-1))/2)



# Assuming your data is stored in a NumPy array called 'data'
data = df.values

# Split the data into training and testing sets
train_size = int(0.8 * len(data))  # 80% for training, 20% for testing
train_data = data[:train_size,:]
test_data = data[train_size:,:]

# Normalize the data (example: Min-Max scaling)
# min_value = np.min(train_data)
# max_value = np.max(train_data)
# train_data = (train_data - min_value) / (max_value - min_value)
# test_data = (test_data - min_value) / (max_value - min_value)

# Create input sequences and targets
def create_sequences(data, timesteps):
    X = []
    y = []
    for i in range(len(data) - timesteps):
        X.append(data[i:i+timesteps,:])
        y.append(data[i+timesteps,:])
    return np.array(X), np.array(y)

timesteps = 5  # Number of previous time steps to use as input

X_train, y_train = create_sequences(train_data, timesteps)
X_test, y_test = create_sequences(test_data, timesteps)

# Convert the data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


print(X_train.shape, y_train.shape)

# plot the data

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(data[:, 0], label='top ellipse X')
plt.plot(data[:, 1], label='top ellipse Y')
plt.plot(data[:, 2], label='bottom ellipse X')
plt.plot(data[:, 3], label='bottom ellipse Y')
plt.legend()
plt.show()


model = HulaHoopLSTM(input_size=4, hidden_size=32, num_layers=2, output_size=4)

model.fit(X_train, y_train, num_epoch=10, batch_size=128)

# evaluate the model
model.net.eval()
y_pred = model.net(X_test)
loss = model.criterion(y_pred, y_test)
print(f'Test loss: {loss.item()}')


