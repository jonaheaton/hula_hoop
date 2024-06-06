import os
import cv2
import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt
import colorsys
import trackpy as tp
import pandas as pd

def display_channels(image):
    # Split the image into its color channels
    channels = cv2.split(image)

    # Create a figure and subplots
    fig, axs = plt.subplots(2, 3, figsize=(20, 20))

    # Display each channel and its histogram
    for i, ax in enumerate(axs[0]):
        ax.imshow(channels[i], cmap='gray')
        ax.set_title(f'Channel {i}')

    for i, ax in enumerate(axs[1]):
        ax.hist(channels[i].ravel(), bins=256, color='gray')
        ax.set_yscale('log')
        ax.set_title(f'Channel {i} Histogram')

    plt.tight_layout()
    plt.show()



def convert_frame_to_rgb(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return frame
    # split the frame into a bottom and top half

def split_frame(frame):
    height, width, _ = frame.shape
    top_half = frame[0:800, :]
    bottom_half = frame[800:, :]

    return top_half, bottom_half



def get_orange_hoop_mask(img):
    # Define the lower and upper boundaries for the color orange in the RGB color space
    lower_orange = np.array([190, 100, 70]) # RGB
    # lower_orange = np.array([180, 80, 50]) # RGB
    upper_orange = np.array([255, 200, 165]) # RGB

    # Create a mask for the color orange
    mask = cv2.inRange(img, lower_orange, upper_orange)

    # Optional: Apply dilation to the mask to remove noise
    kernel = np.ones((3,3), np.uint8)
    # kernel = np.ones((5,5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations = 1)
    return mask
    # Bitwise-AND the mask and the original image
    segmented_image = cv2.bitwise_and(img, img, mask=mask)
    return segmented_image


def get_black_hoop_mask(img, orange_mask,box_sz=300):
    '''
    Given top or bottom half of the image
    find the center of mass of the orange mask
    create a rectangle around the center of mass
    within the rectangle find the black hoop masks
    '''
    center = get_center_of_mass(orange_mask)
    box = get_box_from_center(center, box_sz,box_sz)
    x, y, w, h = box
    roi = img[y:y+h, x:x+w]
    
    black_mask_roi = cv2.inRange(roi, np.array([0,0,0]), np.array([50,50,110]))

    # Create a black mask for the full image
    black_mask_full = np.zeros(img.shape[:2], dtype=np.uint8)
    # black_mask_full = np.zeros_like(img)
    
    # Apply the mask from the ROI to the corresponding location in the full image mask
    black_mask_full[y:y+h, x:x+w] = black_mask_roi

    # Optional: Apply dilation to the mask to remove noise
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.dilate(black_mask_full, kernel, iterations = 1)

    return mask


def get_hoop_markers(img,hoop_mask=None):
    if hoop_mask is None:
        hoop_mask = get_orange_hoop_mask(img)

    black_mask = get_black_hoop_mask(img, hoop_mask)
    black_mask = remove_large_connected_components(black_mask, 300)
    black_mask = remove_small_connected_components(black_mask, 20)
    return black_mask

def get_full_hoop_mask(img):
    '''
    Find the orange mask
    find the black mask
    combine the masks
    '''
    orange_mask = get_orange_hoop_mask(img)
    hoop_markers = get_hoop_markers(img, orange_mask)
    hoop_mask = cv2.bitwise_or(orange_mask, hoop_markers)
    return hoop_mask

def get_center_of_mass(mask):
    '''
    Find the center of mass of the mask
    '''
    moments = cv2.moments(mask)
    cX = int(moments["m10"] / moments["m00"])
    cY = int(moments["m01"] / moments["m00"])
    return (cX, cY)

def get_box_from_center(center, width, height):
    '''
    Create a box of given width and height around the center
    '''
    x = max(0, center[0] - width // 2)
    y = max(0, center[1] - height // 2)
    return (x, y, width, height)

def get_connected_components(mask):
    '''
    Find the connected components of the mask
    '''
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return num_labels, labels, stats, centroids

def get_area_of_mask(mask):
    '''
    Calculate the area of the mask
    '''
    return cv2.countNonZero(mask)


def remove_large_connected_components(mask, max_area):
    '''
    Remove connected components that are larger than the max area
    '''
    num_labels, labels, stats, centroids = get_connected_components(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > max_area:
            mask[labels == i] = 0
    return mask

def remove_small_connected_components(mask, min_area):
    '''
    Remove connected components that are smaller than the min area
    '''
    num_labels, labels, stats, centroids = get_connected_components(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            mask[labels == i] = 0
    return mask


def get_object_centroids(mask):
    '''
    Find the centroids of the connected components in the mask
    '''
    num_labels, labels, stats, centroids = get_connected_components(mask)
    return centroids[1:]

def get_object_centroids_and_areas(mask):
    '''
    Find the centroids and areas of the connected components in the mask
    '''
    num_labels, labels, stats, centroids = get_connected_components(mask)
    return centroids[1:], stats[1:, cv2.CC_STAT_AREA]


def remove_stationay_objects(mask, bad_centroids=None, dist_th=50):
    '''
    Remove connected components that are smaller than the min area and have centroids close to the bad centroids
    '''

    if bad_centroids is None:
        #top half bad centroids
        bad_centroids = [[370.2    , 360.78461538],
                        [379.86419753, 626.51851852]]

    num_labels, labels, stats, centroids = get_connected_components(mask)
    for i in range(1, num_labels):
        # if stats[i, cv2.CC_STAT_AREA] < threshold:
        for bad_centroid in bad_centroids:
            if np.linalg.norm(bad_centroid - centroids[i]) < 50:
                mask[labels == i] = 0
                break

    return mask

def fit_circle_to_disjoint_masks(mask):
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=10, minDist=200, param1=300, param2=0.95, minRadius=80, maxRadius=100)
        # If at least one circle is detected
    if circles is not None:
        circles = np.uint16(np.around(circles))
        return circles[0]
    return None



def fit_ellipse_to_disjoint_masks(mask,min_area=50):
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # remove small contours
    contours = [c for c in contours if cv2.contourArea(c) > min_area]
    # Concatenate all the contours into a single array
    all_points = np.concatenate(contours)

    # Ensure there are at least 5 points (required to fit an ellipse)
    if len(all_points) >= 5:
        # Fit an ellipse to the points
        ellipse = cv2.fitEllipse(all_points)
        return ellipse

    return None


def get_ellipse_mask(image, ellipse, thickness=20):
    ellipse_image = np.zeros_like(image)
    cv2.ellipse(ellipse_image, ellipse, (255, 255, 255), thickness)
    return ellipse_image


def find_near_ellipse_markers(image, ellipse, thickness=20,min_area=20):
    ellipse_image = get_ellipse_mask(image, ellipse, thickness=thickness)
    # Perform a bitwise AND operation between the ellipse image and the original image
    near_ellipse_pixels = cv2.bitwise_and(image, ellipse_image)

    # Convert to HSB color space
    hsv = cv2.cvtColor(near_ellipse_pixels, cv2.COLOR_RGB2HSV_FULL)

    # Create a mask of the pixels within the specified range
    lower_bound = np.array([150, 70, 0])
    # upper_bound = np.array([250,255, 150])
    upper_bound = np.array([200,255, 150])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    # remove small connected components
    mask = remove_small_connected_components(mask, min_area)
    # dilate the mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations = 1)
    return mask



def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.5):
    # Ensure the mask is a grayscale image
    assert len(mask.shape) == 2

    # Create a 3-channel version of the mask
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Set the color of the mask
    mask_color[mask != 0] = color

    # Blend the original image with the color mask
    overlay = cv2.addWeighted(image, 1 - alpha, mask_color, alpha, 0)

    return overlay


def particle_num_to_rgb(num, max_particles):
    '''
    Given a particle number and the max number of particles
    return a color in RGB
    '''
    hue = num / max_particles
    r, g, b = colorsys.hsv_to_rgb(hue, 1, 1)
    return (int(r * 255), int(g * 255), int(b * 255))



def link_top_bottom_markers(top_markers, bottom_markers, top_ellipse, bottom_ellipse, min_dist=50):
    '''
    Given the top and bottom markers
    link the markers based on their distance to the their respective ellipse centers
    '''
    linked_markers = []
    # estimate the radius of the ellipse by taking the square root of the product of the semi-major and semi-minor axes
    top_ellipse_radius = np.sqrt(top_ellipse[1][0]*top_ellipse[1][1])

    # Normalize the distance between the top markers and the ellipse center by the radius of the ellipse
    top_marker_diffs = (top_markers - top_ellipse[0]) / top_ellipse_radius

    bottom_ellipse_radius = np.sqrt(bottom_ellipse[1][0]*bottom_ellipse[1][1])
    bottom_marker_diffs = (bottom_markers - bottom_ellipse[0]) / bottom_ellipse_radius

    # link the top and bottom markers based on their distance to the ellipse centers using trackpy
    top_df = pd.DataFrame(top_marker_diffs, columns=['x', 'y']) 
    top_df['frame'] = 0
    top_df['top_X'] = top_markers[:,0]
    top_df['top_Y'] = top_markers[:,1]

    bottom_df = pd.DataFrame(bottom_marker_diffs, columns=['x', 'y'])
    bottom_df['frame'] = 1
    bottom_df['bottom_X'] = bottom_markers[:,0]
    bottom_df['bottom_Y'] = bottom_markers[:,1]

    # Combine the top and bottom dataframes
    df = pd.concat([top_df, bottom_df])

    # Link the top and bottom markers
    linked_markers = tp.link_df(df, search_range=min_dist, memory=1)

    return linked_markers