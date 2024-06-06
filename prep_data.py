import pickle
import pandas as pd
import numpy as np


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
        data_dict['frame'] = i

        coordinate_dict = raw_data[i]
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


def main(raw_data_path, processed_data_path):

    df = get_hula_hoop_coord(raw_data_path)
    save_csv(df, processed_data_path)


if __name__ == '__main__':

    raw_data_path = '/Users/jonaheaton/Documents/hulahoop_data/output_DSC_7450_data.pt'
    processed_data_path = '/Users/jonaheaton/Documents/hulahoop_data/output_DSC_7450_hula_hoop_data.csv'

    main(raw_data_path, processed_data_path)