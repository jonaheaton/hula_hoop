from utils import *
import json
import argparse
import os
import cv2
import pickle
import pandas as pd

def process_frame(frame):
    """
    Processes a frame of a video to detect and analyze hula hoop markers.

    Args:
        frame: A single frame of a video.

    Returns:
        combined_overlay: A numpy array representing the combined overlay of the top and bottom halves of the frame with the detected markers.
        coordinate_dict: A dictionary containing the coordinates and areas of the detected markers.

    """    
    top_half, bottom_half = split_frame(convert_frame_to_rgb(frame))
    
    top_hoop_mask = get_orange_hoop_mask(top_half)
    bottom_hoop_mask = get_orange_hoop_mask(bottom_half)

    bottom_circle = None
    # bottom_circle = fit_circle_to_disjoint_masks(bottom_hoop_mask)
    bottom_ellipse=fit_ellipse_to_disjoint_masks(bottom_hoop_mask,min_area=20)
    top_ellipse = fit_ellipse_to_disjoint_masks(top_hoop_mask)

    top_overlay = overlay_mask(top_half, top_hoop_mask)
    bottom_overlay = overlay_mask(bottom_half, bottom_hoop_mask)

    if bottom_circle is not None:
        bottom_circle = bottom_circle[0]
        cv2.circle(bottom_overlay, (bottom_circle[0], bottom_circle[1]), bottom_circle[2], (255, 255, 0), 2)
        cv2.circle(bottom_overlay, (bottom_circle[0], bottom_circle[1]), 2, (255, 0, 255), 3)

    if bottom_ellipse is not None:
        cv2.ellipse(bottom_overlay, bottom_ellipse, (255, 255, 0), 2)

    if top_ellipse is not None:
        cv2.ellipse(top_overlay, top_ellipse, (255, 255, 0), 2)
        
    top_markers = find_near_ellipse_markers(top_half, top_ellipse)
    bottom_markers = find_near_ellipse_markers(bottom_half, bottom_ellipse,thickness=10,min_area=5)

    # create a new overlay with the markers
    top_overlay = overlay_mask(top_overlay, top_markers,color=(255,0,0))
    bottom_overlay = overlay_mask(bottom_overlay, bottom_markers,color=(255,0,0))
    

    top_marker_centroids, top_marker_areas = get_object_centroids_and_areas(top_markers)
    bottom_marker_centroids, bottom_marker_areas = get_object_centroids_and_areas(bottom_markers)

    coordinate_dict = {
        'top_ellipse': top_ellipse,
        'bottom_ellipse': bottom_ellipse,
        'top_markers': top_marker_centroids,
        'bottom_markers': bottom_marker_centroids,
        'top_marker_area': top_marker_areas,
        'bottom_marker_area': bottom_marker_areas
    }

    combined_overlay = np.vstack([top_overlay, bottom_overlay])
    return combined_overlay, coordinate_dict




def overlay_ellipse_on_frame(frame, top_ellipse, bottom_ellipse):
    """
    Overlay ellipses on the top and bottom halves of a frame.

    Args:
        frame (numpy.ndarray): The input frame.
        top_ellipse (tuple): A tuple containing the parameters of the top ellipse (center, axes, angle).
        bottom_ellipse (tuple): A tuple containing the parameters of the bottom ellipse (center, axes, angle).

    Returns:
        numpy.ndarray: The frame with the ellipses overlaid on the top and bottom halves.

    """
    top_half, bottom_half = split_frame(convert_frame_to_rgb(frame))
    top_ellipse_mask = get_ellipse_mask(top_half, top_ellipse, thickness=10)
    bottom_ellipse_mask = get_ellipse_mask(bottom_half, bottom_ellipse, thickness=10)
    # convert to binary
    top_ellipse_mask = cv2.cvtColor(top_ellipse_mask, cv2.COLOR_RGB2GRAY)
    bottom_ellipse_mask = cv2.cvtColor(bottom_ellipse_mask, cv2.COLOR_RGB2GRAY)
    top_overlay = overlay_mask(top_half, top_ellipse_mask, color=(0, 255, 0))
    bottom_overlay = overlay_mask(bottom_half, bottom_ellipse_mask, color=(0, 255, 0))
    return np.vstack([top_overlay, bottom_overlay])


def convert_ellipse_info_from_dict(ellipse_dict):
    """
    Convert ellipse information from a dictionary to a tuple format.

    Args:
        ellipse_dict (dict): A dictionary containing ellipse information.

    Returns:
        tuple: A tuple containing the converted ellipse information for the top and bottom ellipses.
            The tuple format is as follows:
            (
                (top_ellipse_X, top_ellipse_Y),
                (top_ellipse_Major_Axis, top_ellipse_Minor_Axis),
                top_ellipse_Angle
            ),
            (
                (bottom_ellipse_X, bottom_ellipse_Y),
                (bottom_ellipse_Major_Axis, bottom_ellipse_Minor_Axis),
                bottom_ellipse_Angle
            )
        Each element in the tuple represents the information for one ellipse.
    """
    top_ellipse_info = [(ellipse_dict['top ellipse X'], ellipse_dict['top ellipse Y']),
                        (ellipse_dict['top ellipse Major Axis'], ellipse_dict['top ellipse Minor Axis']),
                        ellipse_dict['top ellipse Angle']]
    
    bottom_ellipse_info = [(ellipse_dict['bottom ellipse X'], ellipse_dict['bottom ellipse Y']),
                        (ellipse_dict['bottom ellipse Major Axis'], ellipse_dict['bottom ellipse Minor Axis']),
                        ellipse_dict['bottom ellipse Angle']]
    
    return top_ellipse_info, bottom_ellipse_info



def get_ellipse_info_from_dataframe(df, frame_number):
    """
    Retrieves ellipse information from a dataframe for a given frame number.

    Parameters:
    - df (pandas.DataFrame): The dataframe containing ellipse information.
    - frame_number (int): The frame number for which to retrieve the ellipse information.

    Returns:
    - top_ellipse_info (tuple): A tuple containing the information of the top ellipse in the form of:
                                - (x, y): The center coordinates of the top ellipse.
                                - (major_axis, minor_axis): The major and minor axis lengths of the top ellipse.
                                - angle (float): The rotation angle of the top ellipse.
    - bottom_ellipse_info (tuple): A tuple containing the information of the bottom ellipse in the form of:
                                   - (x, y): The center coordinates of the bottom ellipse.
                                   - (major_axis, minor_axis): The major and minor axis lengths of the bottom ellipse.
                                   - angle (float): The rotation angle of the bottom ellipse.
    """
    
    row = df.loc[frame_number]
    top_ellipse_info = [(row['top ellipse X'], row['top ellipse Y']),
                        (row['top ellipse Major Axis'], row['top ellipse Minor Axis']),
                        row['top ellipse Angle']]
    
    bottom_ellipse_info = [(row['bottom ellipse X'], row['bottom ellipse Y']),
                        (row['bottom ellipse Major Axis'], row['bottom ellipse Minor Axis']),
                        row['bottom ellipse Angle']]
    
    return top_ellipse_info, bottom_ellipse_info
                          



def video_in(video_path):
    """
    Opens a video file and returns the video capture object, width, height, and frames per second (fps).

    Parameters:
    video_path (str): The path to the video file.

    Returns:
    cap (cv2.VideoCapture): The video capture object.
    width (int): The width of the video frames.
    height (int): The height of the video frames.
    fps (float): The frames per second of the video.

    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    return cap, width, height, fps

def video_out(output_path, fps, width, height):
    """
    Create a video writer object and return it.

    Parameters:
    output_path (str): The path to the output video file.
    fps (float): Frames per second of the output video.
    width (int): Width of the output video frame.
    height (int): Height of the output video frame.

    Returns:
    cv2.VideoWriter: The video writer object.

    """
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    return out



def slow_down_video(video_path, slow_factor=2, start_frame=0, max_frames=200, output_path=None):
    """
    Slows down a video by a given factor.

    Parameters:
    - video_path (str): The path to the input video file.
    - slow_factor (int): The factor by which to slow down the video. Default is 2.
    - start_frame (int): The starting frame index. Default is 0.
    - max_frames (int): The maximum number of frames to process. Default is 200.
    - output_path (str): The path to save the output video file. If not provided, a default path will be used.

    Returns:
    - output_path (str): The path to the output video file.

    Raises:
    - None

    Example usage:
    video_path = '/path/to/input/video.mp4'
    slow_factor = 2
    start_frame = 0
    max_frames = 200
    output_path = '/path/to/output/video_slow.mp4'

    slow_down_video(video_path, slow_factor, start_frame, max_frames, output_path)
    """
    
    cap, width, height, fps = video_in(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None

    if output_path is None:
        if '.mp4' in video_path:
            mov_ext = '.mp4'
        else:
            mov_ext = '.MOV'
        output_path = video_path.replace(mov_ext, f'_slow_{slow_factor}.mp4')

    out = video_out(output_path, fps / slow_factor, width, height)
    if out is None:
        print("Error: Could not create output video file.")
        cap.release()
        return None

    iter = start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    while iter < start_frame + max_frames:
        # Read the next frame
        ret, frame = cap.read()
        if not ret:
            # We reached the end of the video
            break

        # Write the processed frame to the new video file
        out.write(frame)
        iter += 1

    # Release the video capture and writer objects
    cap.release()
    out.release()

    return output_path




def create_video_with_ellipse_overlay(input_video_path, output_video_path, csv_path, start_frame=None, max_frames=None, slow_factor=1):
    """
    Creates a new video with an ellipse overlay based on the provided CSV file.

    Parameters:
    - input_video_path (str): The path to the input video file.
    - output_video_path (str): The path to save the output video file. If None, the video will not be saved.
    - csv_path (str): The path to the CSV file containing ellipse information.
    - start_frame (int, optional): The starting frame index. If None, the first frame in the CSV file will be used.
    - max_frames (int, optional): The maximum number of frames to process. If None, all frames in the CSV file will be processed.
    - slow_factor (int, optional): The factor by which to slow down the output video. Default is 1 (no slow down).

    Returns:
    - None

    """
    df = pd.read_csv(csv_path, index_col=0)
    cap, width, height, fps = video_in(input_video_path)
    if output_video_path is not None:
        out = video_out(output_video_path, fps/slow_factor, width, height)
    iter = 0

    if start_frame is None:
        start_frame = df.index[0]
    if max_frames is None:
        end_frame = df.index[-1]
        max_frames = end_frame - start_frame

    if iter < start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        iter = start_frame

    while iter < start_frame+max_frames:
        # Read the next frame
        ret, frame = cap.read()
        if not ret:
            # We reached the end of the video
            break

        # Process the frame
        if iter in df.index:
            top_ellipse_info, bottom_ellipse_info = get_ellipse_info_from_dataframe(df, iter)
            overlay = overlay_ellipse_on_frame(frame, top_ellipse_info, bottom_ellipse_info)
            if 'kind' in df.columns:
                desc_text =  df.loc[iter,'kind']
            else:
                desc_text = ''
        else:
            overlay = frame
            desc_text = ''
        processed_frame = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        if desc_text != '':
            cv2.putText(processed_frame, desc_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Write the processed frame to the new video file
        if output_video_path is not None:
            out.write(processed_frame)
        iter += 1

    # Release the video capture and writer objects
    cap.release()
    if output_video_path is not None:
        out.release()


def main_process_video(input_video_path, output_data_path=None, output_video_path=None, start_frame=0, max_frames=200, slow_factor=1):
    """
    Process a video by iterating through its frames, applying a frame processing function, and saving the results.
    The processed video highlights the location of the (orange) hula-hoop on the bottom and top halves of the frame.
    a data file containing the ellipse coordinates and areas is saved as a pickle file.

    Parameters:
    - input_video_path (str): The path to the input video file.
    - output_data_path (str, optional): The path to save the coordinate data as a pickle file. Default is None.
    - output_video_path (str, optional): The path to save the processed video file. Default is None.
    - start_frame (int, optional): The starting frame index. Default is 0.
    - max_frames (int, optional): The maximum number of frames to process. Default is 200.
    - slow_factor (int, optional): The factor by which to slow down the output video. Default is 1.

    Returns:
    - None

    """
    
    # Rest of the code...
def main_process_video(input_video_path, output_data_path=None, output_video_path=None, start_frame=0, max_frames=200,slow_factor=1):

    cap, width, height, fps = video_in(input_video_path)

    if output_video_path is not None:
        out = video_out(output_video_path, fps/slow_factor, width, height)

    coordinate_dict_list = []
    iter = 0


    if iter < start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        iter = start_frame

    while iter < start_frame+max_frames:
        # Read the next frame
        ret, frame = cap.read()
        if not ret:
            # We reached the end of the video
            break

        # Process the frame
        processed_frame, coordinate_dict = process_frame(frame)
        coordinate_dict['frame'] = iter
        coordinate_dict_list.append(coordinate_dict)
        # convert the frame back to BGR
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)

        # Write the processed frame to the new video file
        if output_video_path is not None:
            out.write(processed_frame)
        iter += 1

    cap.release()
    if output_video_path is not None:
        out.release()

    # Save the coordinate data to a pickle file
    if output_data_path is not None:
        with open(output_data_path, 'wb') as f:
            pickle.dump(coordinate_dict_list, f)



if __name__ == '__main__':

    DEFAULT_INPUT_VIDEO = '/Users/jonaheaton/Documents/hulahoop_data/videos/DSC_7450.MOV'
    DEFAULT_INPUT_VIDEO_SLOW = '/Users/jonaheaton/Documents/hulahoop_data/DSC_7450_slow1.MOV'
    DEFAULT_OUTPUT_FILE = '/Users/jonaheaton/Documents/hulahoop_data/output4_DSC_7450_data.pt'
    DEFAULT_OUTPUT_VIDEO = '/Users/jonaheaton/Documents/hulahoop_data/output4_DSC_7450_video.mp4'

    # parse the command line arguments
    parser = argparse.ArgumentParser(description='Process a video file')
    parser.add_argument('-iv', '--input_video', type=str, help='The path to the input video file', default=DEFAULT_INPUT_VIDEO)
    parser.add_argument('-ivs', '--input_video_slow', type=str, help='The path to the slowed input video file', default=DEFAULT_INPUT_VIDEO_SLOW)
    parser.add_argument('-od', '--output_data', type=str,help='The path to the output data file', default=DEFAULT_OUTPUT_FILE)
    parser.add_argument('-ov', '--output_video', type=str, help='The path to the saved output processed video file', default=DEFAULT_OUTPUT_VIDEO)
    parser.add_argument('-sf', '--start_frame', type=int, help='The frame number to start processing from', default=8000)
    parser.add_argument('-mf', '--max_frames', type=int, help='The maximum number of frames to process', default=10)


    args = parser.parse_args()
    slow_factor = 4
    # run the main function

    # main_process_video(args.input_video, args.output_data, args.output_video, args.start_frame, args.max_frames, slow_factor=slow_factor)

    slow_down_video(args.input_video, slow_factor=slow_factor, start_frame=args.start_frame, 
                    max_frames=args.max_frames, output_path=args.input_video_slow)
