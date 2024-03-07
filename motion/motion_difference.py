# Author: jayasingam adhuran

import numpy as np
import cv2
import os
import sys


curr_path = os.path.dirname(os.path.abspath(__file__)) 
if curr_path not in sys.path:
    sys.path.insert(0, curr_path)

from motion_detection import DiffMotionDetector
from motion_detection import MogMotionDetector
from motion_detection import Mog2MotionDetector

def mask_to_tile_pred(pixel_coord, user_data):
    

    coord_tile_list = np.full([pixel_coord.shape[0], pixel_coord.shape[1]], -1)
    for i in range(user_data['config_params']['total_tile_num']):
        
        tile_idx = i

        #if ((tile_idx == (user_data['config_params']['total_tile_num']-2)) or (tile_idx == (user_data['config_params']['total_tile_num']-3))):
        #    continue

        if (tile_idx == 9) or (tile_idx == 10)  or (tile_idx == 11) or (tile_idx == 12)or (tile_idx == 13)  or (tile_idx == 14) or (tile_idx == 15) or (tile_idx == 16):
            continue

        dst_height = user_data['config_params']['converted_height']
        dst_width = user_data['config_params']['converted_width']

        if (tile_idx == 0):
            tile_start_width = 0
            tile_start_height = 0
            tile_width = dst_width
            tile_height = int(dst_height/6)
        
        elif (tile_idx == (user_data['config_params']['total_tile_num']-1)):
            tile_start_width = 0
            tile_start_height = dst_height - int(dst_height/6)
            tile_width = dst_width
            tile_height = int(dst_height/6)
       
        else:
            tile_start_width = ((tile_idx-1) % user_data['config_params']['tile_width_num'])* int(dst_width/4)
            tile_start_height = (tile_idx-1) // user_data['config_params']['tile_width_num'] * int(dst_height/3) + int(dst_height/6)
            tile_width = int(dst_width/4)
            tile_height = int(dst_height/3)
             

        # find coordinates that satisfy both width and height conditions
        hit_coord_mask = pixel_coord[tile_start_height:tile_start_height + tile_height, tile_start_width:tile_start_width + tile_width]#mask_width & mask_height
        hit_coord_mask[hit_coord_mask>0] = tile_idx
        # update coord_tile_list
        coord_tile_list[tile_start_height:tile_start_height + tile_height, tile_start_width:tile_start_width + tile_width] =  hit_coord_mask
    

    return coord_tile_list




def calculate_motion_diff(source_ori, user_data):

    empty_list = []
    unique_tile_list = np.array(empty_list)

    #Open the video file and loading the background image
    video_capture = cv2.VideoCapture(source_ori)
    
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_range = np.arange(start=5, stop=frame_count-5, step=5).tolist()

    for frame_idx in frame_range:

        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = video_capture.read()

        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx-5)
        ret, background_image = video_capture.read()

        #Decalring the diff motion detector object and setting the background
        my_diff_detector = DiffMotionDetector()
        my_diff_detector.setBackground(background_image)


        #Get the mask from the detector objects
        diff_mask = my_diff_detector.returnMask(frame)
    
        #Merge the b/w frame in order to have depth=3
        #diff_mask = cv2.merge([diff_mask, diff_mask, diff_mask])
    
        coord_tile_list = mask_to_tile_pred(diff_mask, user_data)
        coord_tile_list = np.unique(coord_tile_list)
        unique_tile_list = np.append(unique_tile_list, coord_tile_list)


    video_capture.release()

    return unique_tile_list


