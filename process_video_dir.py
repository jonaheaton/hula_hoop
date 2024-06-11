# read multiple 
import os
from read_video import main as main_read_video
from prep_data import main as main_prep_data
from sklearn.preprocessing import StandardScaler, MinMaxScaler

input_dir = '/Users/jonaheaton/Documents/hulahoop_data/videos'
output_dir = '/Users/jonaheaton/Documents/hulahoop_data/output'

datafiles_list = []

for iter,file in enumerate(os.listdir(input_dir)):
    if file.endswith('.MOV'):
        input_video_path = os.path.join(input_dir, file)
        file_name = os.path.splitext(file)[0]

        output_data_path = os.path.join(output_dir, f'{file_name}_data.pt')
        output_video_path = os.path.join(output_dir, f'{file_name}_processed.mp4')
        processed_data_path = os.path.join(output_dir, f'{file_name}_hula_hoop_data.csv')
        if not os.path.exists(processed_data_path):

            main_read_video(input_video_path, output_data_path, output_video_path=output_video_path, start_frame=0, max_frames=10000)
            main_prep_data(output_data_path, processed_data_path)
            datafiles_list.append(processed_data_path)
        else:
            print(f'{processed_data_path} already exists')
            datafiles_list.append(processed_data_path)

    
    if iter > 10:
        break


