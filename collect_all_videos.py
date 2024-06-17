import os
import sys
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
# ! pip install scikit-learn
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from video import main_process_video, slow_down_video, create_video_with_ellipse_overlay
from prep_data import get_hula_hoop_coord, prep_for_lstm, save_csv


def determine_outlier_rows(df, outlier_th=3):

    ellipse_cols = [col for col in df.columns if 'ellipse' in col]
    print(ellipse_cols)
    # df_vals = (df.drop(['frame','video_id','file_name'], axis=1)).values
    df_vals = df[ellipse_cols].values
    df_vals = StandardScaler().fit_transform(df_vals)

    outlier_rows = np.any(np.abs(df_vals) > outlier_th, axis=1)
    print('total number of rows: ', len(outlier_rows))
    print('number of outliers: ', np.sum(outlier_rows))
    df.loc[outlier_rows,ellipse_cols] = np.nan
    return df



def assign_data_types(df):
    # Ensure randomness is reproducible
    np.random.seed(42)
    
    video_ids_ordered_by_size = df['video_id'].value_counts().index.tolist()
    large_videos = video_ids_ordered_by_size[:int(len(video_ids_ordered_by_size)/2)]
    small_videos = video_ids_ordered_by_size[int(len(video_ids_ordered_by_size)/2):]

    # Shuffle the video_id to ensure random selection

    np.random.shuffle(large_videos)
    np.random.shuffle(small_videos)

    # Select video IDs for testing and validation
    test_ids = large_videos[:1] + small_videos[:1]
    validation_ids = large_videos[1:2] + small_videos[1:2]
    training_ids = large_videos[2:] + small_videos[2:]
    print('test_ids: ', test_ids)
    print('validation_ids: ', validation_ids)
    print('training_ids: ', training_ids)
    
    # Initialize the new column with empty strings
    df['data_type'] = ''
    
    # Assign 'test' and 'validation' to the selected video IDs
    df.loc[df['video_id'].isin(test_ids), 'data_type'] = 'test'
    df.loc[df['video_id'].isin(validation_ids), 'data_type'] = 'validation'
    
    # Process the remaining videos for training and validation split
    for video_id in training_ids:
        video_df = df[df['video_id'] == video_id]
        val_split_index = int(len(video_df) * 0.8)  # Calculate the 80% index
        test_split_index = int(len(video_df) * 0.9)  # Calculate the 90% index
        # Assign 'training' to the first 70% of rows
        df.loc[video_df.index[:val_split_index], 'data_type'] = 'training'
        # Assign 'validation' to the last 30% of rows
        df.loc[video_df.index[val_split_index:test_split_index], 'data_type'] = 'validation'
        # Assign 'test' to the last 50% of rows
        df.loc[video_df.index[test_split_index:], 'data_type'] = 'test'
    
    return df


######## Main Script ########

input_data_dir = '/Users/jonaheaton/Documents/hulahoop_data/videos'
input_data_list = os.listdir(input_data_dir)

output_data_dir = '/Users/jonaheaton/Documents/hulahoop_data/output_data'
os.makedirs(output_data_dir, exist_ok=True)

all_ellipse_data_file = '/Users/jonaheaton/Documents/hulahoop_data/all_ellipse_data.csv'


all_ellipse_data_list = []

for input_file in input_data_list:
    input_file_path = os.path.join(input_data_dir, input_file)
    output_file_path = os.path.join(output_data_dir, input_file.replace('.MOV', '.pkl'))

    video_id = input_file.split('_')[1].replace('.MOV', '')

    if os.path.exists(output_file_path):
        print(f'{output_file_path} already exists. Skipping...')
        
    else:
        print(f'Processing {input_file_path}')

        try:
            main_process_video(input_video_path=input_file_path, 
                        output_data_path=output_file_path, 
                        output_video_path=None, 
                        start_frame=0, 
                        max_frames=10000,
                        slow_factor=1)
        except Exception as e:
            print(f'Error processing {input_file_path}: {e}')

            # Save error information to a log file
            with open('/Users/jonaheaton/Documents/hulahoop_data/error_log.txt', 'a') as f:
                f.write(f'Error processing {input_file_path}: {e}\n')
            continue
    

    ellipse_df = get_hula_hoop_coord(output_file_path)
    ellipse_df['video_id'] = video_id
    ellipse_df['file_name'] = input_file
    all_ellipse_data_list.append(ellipse_df)


all_ellipse_data = pd.concat(all_ellipse_data_list)

all_ellipse_data = assign_data_types(all_ellipse_data)

print(all_ellipse_data['data_type'].value_counts())

all_ellipse_data = determine_outlier_rows(all_ellipse_data, outlier_th=3)

save_csv(all_ellipse_data, all_ellipse_data_file)



 