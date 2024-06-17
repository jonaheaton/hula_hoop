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
save_csv(all_ellipse_data, all_ellipse_data_file)


 