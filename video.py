from utils import *
import json
import argparse
import os
import cv2
import pickle
import pandas as pd

def process_frame(frame):
    
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
    top_half, bottom_half = split_frame(convert_frame_to_rgb(frame))
    top_ellipse_mask = get_ellipse_mask(top_half, top_ellipse,thickness=10)
    bottom_ellipse_mask = get_ellipse_mask(bottom_half, bottom_ellipse,thickness=10)
    # convert to binary
    top_ellipse_mask = cv2.cvtColor(top_ellipse_mask, cv2.COLOR_RGB2GRAY)
    bottom_ellipse_mask = cv2.cvtColor(bottom_ellipse_mask, cv2.COLOR_RGB2GRAY)
    top_overlay = overlay_mask(top_half, top_ellipse_mask, color=(0,255,0))
    bottom_overlay = overlay_mask(bottom_half, bottom_ellipse_mask, color=(0,255,0))
    return np.vstack([top_overlay, bottom_overlay])


def convert_ellipse_info_from_dict(ellipse_dict):

    top_ellipse_info = [(ellipse_dict['top ellipse X'], ellipse_dict['top ellipse Y']),
                        (ellipse_dict['top ellipse Major Axis'], ellipse_dict['top ellipse Minor Axis']),
                        ellipse_dict['top ellipse Angle']]
    
    bottom_ellipse_info = [(ellipse_dict['bottom ellipse X'], ellipse_dict['bottom ellipse Y']),
                        (ellipse_dict['bottom ellipse Major Axis'], ellipse_dict['bottom ellipse Minor Axis']),
                        ellipse_dict['bottom ellipse Angle']]
    
    return top_ellipse_info, bottom_ellipse_info

def get_ellipse_info_from_dataframe(df, frame_number):
    row = df.loc[frame_number]
    top_ellipse_info = [(row['top ellipse X'], row['top ellipse Y']),
                        (row['top ellipse Major Axis'], row['top ellipse Minor Axis']),
                        row['top ellipse Angle']]
    
    bottom_ellipse_info = [(row['bottom ellipse X'], row['bottom ellipse Y']),
                        (row['bottom ellipse Major Axis'], row['bottom ellipse Minor Axis']),
                        row['bottom ellipse Angle']]
    
    return top_ellipse_info, bottom_ellipse_info
                          



def video_in(video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    return cap, width, height, fps

def video_out(output_path, fps, width, height):
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    return out

def slow_down_video(video_path, slow_factor=2, start_frame=0, max_frames=200, output_path=None):
    cap, width, height, fps = video_in(video_path)
    if '.mp4' in video_path:
        mov_ext = '.mp4'
    else:
        mov_ext = '.MOV'
    if output_path is None:
        output_path = video_path.replace(mov_ext, f'_slow_{slow_factor}.mp4')

    out = video_out(output_path, fps/slow_factor, width, height)

    iter = 0
    try:
        if iter < start_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            iter = start_frame

        while iter < start_frame+max_frames:
            # Read the next frame
            ret, frame = cap.read()
            if not ret:
                # We reached the end of the video
                break

            # Write the processed frame to the new video file
            out.write(frame)
            iter += 1

    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"Processed {iter} frames")

    finally:
        # Release the video capture and writer objects
        cap.release()
        out.release()

    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     for i in range(slow_factor):
    #         out.write(frame)
    # cap.release()
    # out.release()

    return output_path


def create_video_with_ellipse_overlay(input_video_path, output_video_path, csv_path,start_frame=None,max_frames=None,slow_factor=1):
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

    try:
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
            # You can add code here to do something with the processed frame

    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"Processed {iter} frames")

    finally:
        # Release the video capture and writer objects
        cap.release()
        if output_video_path is not None:
            out.release()


def main_process_video(input_video_path, output_data_path=None, output_video_path=None, start_frame=0, max_frames=200,slow_factor=1):

    cap, width, height, fps = video_in(input_video_path)

    if output_video_path is not None:
        out = video_out(output_video_path, fps/slow_factor, width, height)

    coordinate_dict_list = []
    iter = 0

    try:
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
            # You can add code here to do something with the processed frame

    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"Processed {iter} frames")

    finally:
        # Release the video capture and writer objects
        cap.release()
        if output_video_path is not None:
            out.release()

    # Save the coordinate data to a pickle file
    if output_data_path is not None:
        with open(output_data_path, 'wb') as f:
            pickle.dump(coordinate_dict_list, f)



if __name__ == '__main__':

    DEFAULT_INPUT_VIDEO = '/Users/jonaheaton/Documents/hulahoop_data/videos/DSC_7450.MOV'
    DEFAULT_INPUT_VIDEO_SLOW = '/Users/jonaheaton/Documents/hulahoop_data/DSC_7450_slow.MOV'
    DEFAULT_OUTPUT_FILE = '/Users/jonaheaton/Documents/hulahoop_data/output4_DSC_7450_data.pt'
    DEFAULT_OUTPUT_VIDEO = '/Users/jonaheaton/Documents/hulahoop_data/output4_DSC_7450_video.mp4'

    # parse the command line arguments
    parser = argparse.ArgumentParser(description='Process a video file')
    parser.add_argument('-iv', '--input_video', type=str, help='The path to the input video file', default=DEFAULT_INPUT_VIDEO)
    parser.add_argument('-ivs', '--input_video_slow', type=str, help='The path to the slowed input video file', default=DEFAULT_INPUT_VIDEO_SLOW)
    parser.add_argument('-od', '--output_data', type=str,help='The path to the output data file', default=DEFAULT_OUTPUT_FILE)
    parser.add_argument('-ov', '--output_video', type=str, help='The path to the saved output processed video file', default=DEFAULT_OUTPUT_VIDEO)
    parser.add_argument('-sf', '--start_frame', type=int, help='The frame number to start processing from', default=1892)
    parser.add_argument('-mf', '--max_frames', type=int, help='The maximum number of frames to process', default=400)


    args = parser.parse_args()
    slow_factor = 4
    # run the main function

    main_process_video(args.input_video, args.output_data, args.output_video, args.start_frame, args.max_frames, slow_factor=slow_factor)

    slow_down_video(args.input_video, slow_factor=slow_factor, start_frame=args.start_frame, 
                    max_frames=args.max_frames, output_file=args.input_video_slow)
