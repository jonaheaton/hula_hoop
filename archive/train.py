# %%
import pandas as pd
import numpy as np
from model import HulaHoopLSTM
import torch
import matplotlib.pyplot as plt
# ! pip install scikit-learn
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils import create_sequences, get_data_scaler


train_frac = 0.5

input_data_file = '/Users/jonaheaton/Documents/hulahoop_data/output_DSC_7450_hula_hoop_data.csv'
output_dir = '/Users/jonaheaton/Documents/hulahoop_data/output_7450'
os.makedirs(output_dir, exist_ok=True)
df = pd.read_csv(input_data_file)

# %%
coor_vars = ['top ellipse X','top ellipse Y','bottom ellipse X','bottom ellipse Y']
axis_vars = ['top ellipse Major Axis','top ellipse Minor Axis','bottom ellipse Major Axis','bottom ellipse Minor Axis']
angle_vars  = ['top ellipse Angle','bottom ellipse Angle']
frame_list = df['frame'].values
df =df[coor_vars + axis_vars + angle_vars]

# look for outliers and replace them with the average of the previous and next values
for col in df.columns:
    outlier_th = df[col].quantile(0.95) - df[col].quantile(0.05)
    print(outlier_th)
    # if col in coor_vars:
    #     outlier_th = 1000
    # elif col in axis_vars:
    #     outlier_th = 1000
    # elif col in angle_vars:
    #     outlier_th = 200
    df[col] = df[col].mask((df[col].diff().abs() > outlier_th) & (df[col].diff(-1).abs() > outlier_th), np.nan)
    print(col, df[col].isna().sum())
    df[col] = df[col].fillna((df[col].shift(1) + df[col].shift(-1))/2)

# print how many missing values are there
print(df.isna().sum())



# %%
# plot the data
train_size = int(train_frac * df.shape[0])  # 70% for training, 30% for testing
fig, axs = plt.subplots(5, 2, figsize=(22, 14))
for i, col in enumerate(df.columns):
    ax = axs[i//2, i%2]
    ax.plot(frame_list[:train_size], df[col].values[:train_size], label='Train')
    ax.plot(frame_list[train_size:], df[col].values[train_size:], label='Test')
    ax.set_title(col)

# add space between subplots
plt.subplots_adjust(hspace=0.4)
plt.savefig(os.path.join(output_dir, 'data_plot.png'), dpi=300, bbox_inches='tight')


# %%
# plot the data
train_size = int(train_frac * df.shape[0])  # 70% for training, 30% for testing
fig, axs = plt.subplots(5, 2, figsize=(22, 14))
for i, col in enumerate(df.columns):
    ax = axs[i//2, i%2]
    ax.plot(frame_list[:train_size], df[col].values[:train_size], label='Train')
    ax.plot(frame_list[train_size:], df[col].values[train_size:], label='Test')
    ax.set_title(col)
    ax.set_xlim([frame_list[train_size-100], frame_list[train_size+100]])

# add space between subplots
plt.subplots_adjust(hspace=0.4)
plt.savefig(os.path.join(output_dir, 'zoom_data_plot.png'), dpi=300, bbox_inches='tight')


# %% Split the data into training and testing sets
data = df.values

# Split the data into training and testing sets
train_size = int(train_frac * len(data))  # 70% for training, 30% for testing
train_data = data[:train_size,:]
test_data = data[train_size:,:]
train_frames = frame_list[:train_size]
test_frames = frame_list[train_size:]
# %%

# Normalize the data (example: Min-Max scaling)
scaler = get_data_scaler(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

# %%
# Create input sequences and targets
timesteps = 5  # Number of previous time steps to use as input

X_train, y_train = create_sequences(train_data, timesteps)
X_test, y_test = create_sequences(test_data, timesteps)


# drop the data with nan values
nan_idx = np.isnan(X_train).any(axis=1).any(axis=1)
X_train = X_train[~nan_idx,:]
y_train = y_train[~nan_idx]

nan_idx = np.isnan(y_train).any(axis=1)
X_train = X_train[~nan_idx,:]
y_train = y_train[~nan_idx]

nan_idx = np.isnan(X_test).any(axis=1).any(axis=1)
X_test = X_test[~nan_idx,:]
y_test = y_test[~nan_idx]

nan_idx = np.isnan(y_test).any(axis=1)
X_test = X_test[~nan_idx,:]
y_test = y_test[~nan_idx]

# Convert the data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# %%

# create the model and fit
model = HulaHoopLSTM(input_size=10, hidden_size=64, output_size=10, num_layers=3)

model.fit(X_train, y_train, num_epoch=100, batch_size=64)


# %%
# evaluate the model
y_pred = model.predict(X_test)
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}')

# Save the predicted results
y_pred_unscaled = scaler.inverse_transform(y_pred)
y_test_unscaled = scaler.inverse_transform(y_test)
y_train_unscaled = scaler.inverse_transform(y_train)

df_train = pd.DataFrame(y_train_unscaled, columns=df.columns, index=range(timesteps, len(y_train_unscaled)+timesteps))
df_pred = pd.DataFrame(y_pred_unscaled, columns=df.columns, index=range(train_size+timesteps, train_size+len(y_pred_unscaled)+timesteps))
df_test = pd.DataFrame(y_test_unscaled, columns=df.columns, index=range(train_size+timesteps, train_size+len(y_test)+timesteps))

df_train = pd.DataFrame(y_train_unscaled, columns=df.columns, index=train_frames[timesteps:len(y_train)+timesteps])
df_pred = pd.DataFrame(y_pred_unscaled, columns=df.columns, index=test_frames[timesteps:len(y_pred)+timesteps])
df_test = pd.DataFrame(y_test_unscaled, columns=df.columns, index=test_frames[timesteps:len(y_test)+timesteps])

# df_pred.to_csv(os.path.join(output_dir, 'predicted_data.csv'), index=True)
# df_test.to_csv(os.path.join(output_dir, 'test_data.csv'), index=True)
# df_train.to_csv(os.path.join(output_dir, 'train_data.csv'), index=True)

# %%

# predict by extrapolating the last 5 frames
predictions = []
last_data = X_train[-1,:,:]
last_timepoint = train_frames[-1]
num_pred = 200

for i in range(num_pred):
    last_data_tensor = torch.tensor(last_data, dtype=torch.float32).unsqueeze(0)
    pred = model.predict(last_data_tensor)
    predictions.append(pred)
    # last_data = np.concatenate([last_data[1:], pred[0]]) 
    last_data = np.concatenate([last_data[1:], np.expand_dims(pred[0], axis=0)])

predictions = np.array(predictions).squeeze()
predictions_unscaled = scaler.inverse_transform(predictions)

df_pred_extrapolate = pd.DataFrame(predictions_unscaled, columns=df.columns, index=range(last_timepoint, last_timepoint+num_pred))
df_pred_extrapolate.to_csv(os.path.join(output_dir, 'predicted_data_extrapolate.csv'), index=True)

# %%
