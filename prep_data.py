import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import create_sequences, get_data_scaler
import torch

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    
def save_pickle(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def save_csv(data, file_path):
    data.to_csv(file_path, index=False)

def get_hula_hoop_coord(raw_data_path):

    raw_data = load_pickle(raw_data_path)
    
    data_dict_list = []
    for i in range(len(raw_data)):
        data_dict = {}
        # if 'frame' not in raw_data[i]:
        #     data_dict['frame'] = i

        coordinate_dict = raw_data[i]
        data_dict['frame'] = coordinate_dict['frame']
        top_ellipse = coordinate_dict['top_ellipse']
        bottom_ellipse = coordinate_dict['bottom_ellipse']
        data_dict['top ellipse X'] = top_ellipse[0][0]
        data_dict['top ellipse Y'] = top_ellipse[0][1]
        data_dict['top ellipse Major Axis'] = top_ellipse[1][0]
        data_dict['top ellipse Minor Axis'] = top_ellipse[1][1]
        data_dict['top ellipse Angle'] = top_ellipse[2]
        data_dict['bottom ellipse X'] = bottom_ellipse[0][0]
        data_dict['bottom ellipse Y'] = bottom_ellipse[0][1]
        data_dict['bottom ellipse Major Axis'] = bottom_ellipse[1][0]
        data_dict['bottom ellipse Minor Axis'] = bottom_ellipse[1][1]
        data_dict['bottom ellipse Angle'] = bottom_ellipse[2]

        data_dict_list.append(data_dict)

    df = pd.DataFrame(data_dict_list)

    return df


# prep for lstm is now defunct, don't use it
def prep_for_lstm(input_df,
                    yes_plot=False,
                    train_frac=0.7,
                    output_dir=None,
                    lookback_window=5,
                    yes_scale_data=False,
                    verbose=False,
                    as_tensor=False):
    coor_vars = ['top ellipse X','top ellipse Y','bottom ellipse X','bottom ellipse Y']
    axis_vars = ['top ellipse Major Axis','top ellipse Minor Axis','bottom ellipse Major Axis','bottom ellipse Minor Axis']
    angle_vars  = ['top ellipse Angle','bottom ellipse Angle']
    frame_list = input_df['frame'].values
    df =input_df[coor_vars + axis_vars + angle_vars].copy()


    # look for outliers and replace them with the average of the previous and next values
    for col in df.columns:
        outlier_th = df[col].quantile(0.95) - df[col].quantile(0.05)
        # print(outlier_th)
        df[col] = df[col].mask((df[col].diff().abs() > outlier_th) & (df[col].diff(-1).abs() > outlier_th), np.nan)
        if verbose: print(f'{col} (th={outlier_th:.3f}), # nans=' , df[col].isna().sum())
        df[col] = df[col].fillna((df[col].shift(1) + df[col].shift(-1))/2)

    # print how many missing values are there
    print(df.isna().sum())

    if yes_plot:
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
        if output_dir is not None:
            plt.savefig(os.path.join(output_dir, 'data_plot.png'), dpi=300, bbox_inches='tight')
            plt.close()

        # %%
        # plot the data
        if len(frame_list[train_size:]) > 200:
            fig, axs = plt.subplots(5, 2, figsize=(22, 14))
            for i, col in enumerate(df.columns):
                ax = axs[i//2, i%2]
                ax.plot(frame_list[:train_size], df[col].values[:train_size], label='Train')
                ax.plot(frame_list[train_size:], df[col].values[train_size:], label='Test')
                ax.set_title(col)
                ax.set_xlim([frame_list[train_size-100], frame_list[train_size+100]])

            # add space between subplots
            plt.subplots_adjust(hspace=0.4)
            if output_dir is not None:
                plt.savefig(os.path.join(output_dir, 'zoom_data_plot.png'), dpi=300, bbox_inches='tight')
                plt.close()


    # %% Split the data into training and testing sets
    data = df.values
    train_size = int(train_frac * len(data))  # 70% for training, 30% for testing
    train_data = data[:train_size,:]
    test_data = data[train_size:,:]
    train_frames = frame_list[:train_size]
    test_frames = frame_list[train_size:]

    # Normalize the data (example: Min-Max scaling)
    if yes_scale_data:
        scaler = get_data_scaler(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)
    else:
        scaler = None


    X_train, y_train = create_sequences(train_data, lookback_window)
    X_test, y_test = create_sequences(test_data, lookback_window)
    train_frames = train_frames[lookback_window:]
    test_frames = test_frames[lookback_window:]

    # drop the data with nan values
    nan_idx = np.isnan(X_train).any(axis=1).any(axis=1)
    X_train = X_train[~nan_idx,:]
    y_train = y_train[~nan_idx]
    train_frames = train_frames[~nan_idx]

    nan_idx = np.isnan(y_train).any(axis=1)
    X_train = X_train[~nan_idx,:]
    y_train = y_train[~nan_idx]
    train_frames = train_frames[~nan_idx]

    nan_idx = np.isnan(X_test).any(axis=1).any(axis=1)
    X_test = X_test[~nan_idx,:]
    y_test = y_test[~nan_idx]
    test_frames = test_frames[~nan_idx]

    nan_idx = np.isnan(y_test).any(axis=1)
    X_test = X_test[~nan_idx,:]
    y_test = y_test[~nan_idx]
    test_frames = test_frames[~nan_idx]

    # Convert the data to PyTorch tensors
    if as_tensor:
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

    train_dict = {
        'X': X_train,
        'y': y_train,
        'frames': train_frames,
        'col names': df.columns
    }

    test_dict = {
        'X': X_test,
        'y': y_test,
        'frames': test_frames,
        'col names': df.columns
    }

    return train_dict, test_dict, scaler


def main(raw_data_path, processed_data_path):

    df = get_hula_hoop_coord(raw_data_path)
    save_csv(df, processed_data_path)


if __name__ == '__main__':

    raw_data_path = '/Users/jonaheaton/Documents/hulahoop_data/output_DSC_7450_data.pt'
    processed_data_path = '/Users/jonaheaton/Documents/hulahoop_data/output_DSC_7450_hula_hoop_data.csv'

    main(raw_data_path, processed_data_path)