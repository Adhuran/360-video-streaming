# Author: jayasingam adhuran

import os
import cv2
import yaml
import argparse
import shutil
import numpy as np
import json
from copy import deepcopy
import subprocess
from pathlib import Path

from e3po.utils import get_logger
from e3po.utils.data_utilities import transcode_video, segment_video, resize_video
from e3po.utils.decision_utilities import predict_motion_tile, tile_decision, generate_dl_list
from e3po.utils.projection_utilities import fov_to_3d_polar_coord, \
    _3d_polar_coord_to_pixel_coord, pixel_coord_to_tile, pixel_coord_to_relative_tile_coord

from e3po.utils.json import write_video_json, write_decision_json
from e3po.utils.data_utilities import get_video_size, remove_temp_files, remove_temp_video
import os.path as osp
from .yuv import yuv_import8, yuv_export

from .custom_projection_utilities import pixel_coord_to_tile_corrected, pixel_coord_to_relative_tile_coord_corrected, transform_projection_, pixel_coord_to_tile_corrected_decision_making

from .viewport.Viewport import viewport_prediction_tester
from .motion.motion_difference import calculate_motion_diff
from e3po.utils.evaluation_utilities import extract_frame

from .ffprobe import run_ffprobe
from .calulate_wsmse import calculate_psnr_ws


def findRepeating(arr, size):
    
    frequency = {}
    
    for i in range (0, size):
        frequency[arr[i]] = \
        frequency.get(arr[i], 0) + 1
    return frequency

def pixel_coord_to_tile_pred(pixel_coord, total_tile_num, user_data):
    

    coord_tile_list = np.full(pixel_coord[0].shape, -1)
    for i in range(total_tile_num):
        
        tile_idx = i
        
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
        

        #index_width = tile_idx % user_data['config_params']['tile_width_num']        # determine which col
        #index_height = tile_idx // user_data['config_params']['tile_width_num']      # determine which row


        #tile_start_width = index_width * user_data['config_params']['tile_width']
        #tile_start_height = index_height * user_data['config_params']['tile_height']
        #tile_width = user_data['config_params']['tile_width']
        #tile_height = user_data['config_params']['tile_height']

        # Create a Boolean mask to check if the coordinates are within the tile range
        mask_width = (tile_start_width <= pixel_coord[0]) & (pixel_coord[0] < tile_start_width + tile_width)
        mask_height = (tile_start_height <= pixel_coord[1]) & (pixel_coord[1] < tile_start_height + tile_height)

        # find coordinates that satisfy both width and height conditions
        hit_coord_mask = mask_width & mask_height

        # update coord_tile_list
        coord_tile_list[hit_coord_mask] = tile_idx
    
    coord_tile_list
    return coord_tile_list


def get_opt_(video_stream, source_video_uri):
    """
    Get options.
    Read command line parameters, read configuration file parameters, and initialize logger.

    Returns
    -------
    dict
        Configurations.

    Examples
    --------
    >> opt = get_opt()
    """
    # Read the command line input parameter.
    parser = argparse.ArgumentParser()
    parser.add_argument('-approach_name', type=str, required=True,
                        help="test approach name")
    parser.add_argument('-approach_type', type=str, required=True,
                        help="approach type")
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.abspath(__file__)).split('approaches')[0]
    opt_path = 'e3po.yml'
    try:
        
        if os.path.exists(os.path.join(project_dir, opt_path)):
            with open(project_dir + opt_path, 'r', encoding='UTF-8') as f:
                opt = yaml.safe_load(f.read())

            opt['approach_name'] = args.approach_name
            opt['approach_type'] = args.approach_type
            opt['project_path'] = project_dir[:-1]

            if not opt['e3po_settings']['video']['origin']['video_dir']:
                opt['e3po_settings']['video']['origin']['video_dir'] = osp.join(project_dir[:-1], 'source', 'video')
    
        else:
            opt ={'e3po_settings':{'encoding_params':{}, 'metric':{}}}
            opt['e3po_settings']['encoding_params']['video_fps'] = video_stream['avg_frame_rate']
            opt['e3po_settings']['encoding_params']['qp_list'] = [29]
            opt['e3po_settings']['metric']['gc_w1'] = 0.09
            opt['e3po_settings']['metric']['gc_alpha'] = 0.006
            opt['e3po_settings']['metric']['gc_beta'] = 10
            opt['e3po_settings']['metric']['range_fov'] = [ 89, 89 ]
            opt['e3po_settings']['metric']['fov_resolution'] = [ 1920, 1832 ]
    except:
        tmp_cap = cv2.VideoCapture()
        assert tmp_cap.open(source_video_uri), f"[error] Can't read video[{source_video_uri}]"
        frame_rate = tmp_cap.get(cv2.CAP_PROP_FPS)
        opt ={'e3po_settings':{'encoding_params':{}, 'metric':{}}}
        opt['e3po_settings']['encoding_params']['video_fps'] = frame_rate
        opt['e3po_settings']['encoding_params']['qp_list'] = [29]
        opt['e3po_settings']['metric']['gc_w1'] = 0.09
        opt['e3po_settings']['metric']['gc_alpha'] = 0.006
        opt['e3po_settings']['metric']['gc_beta'] = 10
        opt['e3po_settings']['metric']['range_fov'] = [ 89, 89 ]
        opt['e3po_settings']['metric']['fov_resolution'] = [ 1920, 1832 ]
    
    return opt

def tile_prediction_chunk(source_video_uri, user_data):
   
    unique_tile_list = calculate_motion_diff(source_video_uri, user_data)
    tiles_predicted = findRepeating(unique_tile_list, len(unique_tile_list))
    selected_tile_list = np.unique(unique_tile_list).tolist()#np.arange(start=1, stop=9, step=1).tolist()#
    #selected_tile_list = [i for i in tiles_predicted if tiles_predicted[i]>1]
    #curr_path = os.path.dirname(os.path.abspath(__file__)) 
    #json_path=os.path.join(curr_path,"tile_prediction.json")
    #write_tile_prediction_json(json_path, user_data['chunk_idx'], selected_tile_list)

    #selected_tile_list = np.arange(start=1, stop=17, step=1).tolist()
    return selected_tile_list


def viewport_prediction_chunk(source_video_uri, src_proj, dst_proj, src_resolution, dst_resolution, dst_video_folder, chunk_info, config_params, user_data):
    
    try:
        tmp_cap = cv2.VideoCapture()
        assert tmp_cap.open(source_video_uri), f"[error] Can't read video[{source_video_uri}]"
        src_video_h = src_resolution[0]
        src_video_w = src_resolution[1]
        ffmpeg_settings = config_params['ffmpeg_settings']
        curr_fov = user_data['opt']['e3po_settings']['metric']
   
        source_video_uri_yuv = source_video_uri.split('mp4')[0] + 'yuv'
        cmd = f"{ffmpeg_settings['ffmpeg_path']} " \
              f"-i {source_video_uri} " \
              f"-r 30 " \
              f"-pix_fmt yuv420p " \
              f"-s {src_video_w}x{src_video_h} " \
              f"-f rawvideo " \
              f"-y {source_video_uri_yuv} " \
              f"-loglevel {ffmpeg_settings['loglevel']}"
        get_logger().debug(cmd)
        os.system(cmd)
    
        frame_count = int(tmp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = 32
        if frame_interval > frame_count:
            frame_interval = 1
        sequence_data = {'src':source_video_uri_yuv, 'res':[src_video_w, src_video_h], 'frame_count':frame_count}
        predictions = viewport_prediction_tester(sequence_data, frame_interval = frame_interval)
        empty_list = []
        unique_tile_list = np.array(empty_list)
        frame_tested = len(predictions)
        for i in range (frame_tested):
            pred_coord = predictions[i]['Predicitions'].numpy()
            predicitions_lat = [pred_coord[0].item(), pred_coord[2].item(), pred_coord[4].item()]#, pred_coord[6].item()]
            predicitions_lon = [pred_coord[1].item(), pred_coord[3].item(), pred_coord[5].item()]#, pred_coord[7].item()]
            for j in range (len(predicitions_lat)):
                # calculating fov_uv parameters
                fov_ypr = [float(2*np.pi*predicitions_lon[j]/360.0 + np.pi), float(np.pi*predicitions_lat[j]/180.0), 0]
                _3d_polar_coord = fov_to_3d_polar_coord(fov_ypr, curr_fov['range_fov'], curr_fov['fov_resolution'])
                pixel_coord = _3d_polar_coord_to_pixel_coord(_3d_polar_coord, config_params['projection_mode'], [config_params['converted_height'], config_params['converted_width']])
                coord_tile_list = pixel_coord_to_tile_pred(pixel_coord, config_params['total_tile_num'], user_data)
                coord_tile_list = np.unique(coord_tile_list)
                unique_tile_list = np.append(unique_tile_list, coord_tile_list)
         
        remove_temp_video(source_video_uri_yuv)
        tiles_predicted = findRepeating(unique_tile_list, len(unique_tile_list))
        selected_tile_list = np.unique(unique_tile_list).tolist()#np.arange(start=1, stop=9, step=1).tolist()#
        #selected_tile_list = [i for i in tiles_predicted if tiles_predicted[i]>1]

        selected_tiles_salient =tile_prediction_chunk(source_video_uri, user_data)
        two_prediction_tiles = selected_tile_list + selected_tiles_salient
        selected_tile_list = np.unique(two_prediction_tiles).tolist()


        curr_path = os.path.dirname(os.path.abspath(__file__)) 
        json_path=os.path.join(curr_path,"tile_prediction.json")
        write_tile_prediction_json(json_path, user_data['chunk_idx'], selected_tile_list)

        selected_tile_list = np.arange(start=1, stop=17, step=1).tolist()
    except:
        selected_tile_list = np.arange(start=1, stop=9, step=1).tolist()#
        
        curr_path = os.path.dirname(os.path.abspath(__file__)) 
        json_path=os.path.join(curr_path,"tile_prediction.json")
        write_tile_prediction_json(json_path, user_data['chunk_idx'], selected_tile_list)

        selected_tile_list = np.arange(start=1, stop=17, step=1).tolist()
    return selected_tile_list


def get_rd_analysis(ffmpeg_settings, source_video_uri, dst_video_folder, user_data, chunk_info):
   
    encoding_params = user_data['opt']['e3po_settings']['encoding_params']
      
    src_resolution = [user_data['video_info']['height'], user_data['video_info']['width']]
    src_projection = user_data['config_params']['projection_mode']
    bg_resolution = [user_data['config_params']['background_height'], user_data['config_params']['background_width']]
    bg_projection = user_data['config_params']['background_info']['background_projection_mode']      

    if user_data["video_info"]["video_stream"]['pix_fmt'] == 'yuv420p':
        bg_video_uri = transcode_video_(
        source_video_uri, src_projection, bg_projection, src_resolution, bg_resolution,
        dst_video_folder, chunk_info, user_data['config_params']['ffmpeg_settings'], user_data
    )
    else:
        bg_video_uri = transcode_video(
    source_video_uri, src_projection, bg_projection, src_resolution, bg_resolution,
    dst_video_folder, chunk_info, config_params['ffmpeg_settings']
    )                

    bg_video_size = get_video_size(bg_video_uri)
    remove_temp_video(bg_video_uri)

    result_video_name = "test"
    dst_video_uri = osp.join(dst_video_folder, f'{result_video_name}.mp4')
    #os.chdir(dst_video_folder)
    qp_val = encoding_params['qp_list'][0]
    try: 
        user_data["video_info"]["video_stream"]['color_primaries']
        color_primaries = user_data["video_info"]["video_stream"]['color_primaries']
        color_prim_cmd = 'colorprim='+color_primaries+':transfer='+color_primaries+':colormatrix='+color_primaries
    except:
        color_prim_cmd = ''
    
    tmp_cap = cv2.VideoCapture()
    assert tmp_cap.open(source_video_uri), f"[error] Can't read video[{source_video_uri}]"
    frame_count = int(tmp_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    best_qp = encoding_params['qp_list'][0]
    init_qp = encoding_params['qp_list'][0]
   
    best_gc = 0.0
    qp_list = [init_qp + 2, init_qp + 1, init_qp, init_qp - 1, init_qp - 2]
      

    w_1 = user_data['opt']['e3po_settings']['metric']['gc_w1']
    alpha = user_data['opt']['e3po_settings']['metric']['gc_alpha']
    beta = user_data['opt']['e3po_settings']['metric']['gc_beta']

    for test_qp in qp_list:
     
        cmd = f"{ffmpeg_settings['ffmpeg_path']} " \
              f"-i {source_video_uri} " \
              f"-r {encoding_params['video_fps']} " \
              f"-threads {2} " \
              f"-c:v libx264  " \
              f"-preset slower " \
              f" -x264-params \"{color_prim_cmd}\" " \
              f"-r {encoding_params['video_fps']} " \
              f"-qp {test_qp} " \
              f"-y {dst_video_uri} " \
              f"-loglevel {ffmpeg_settings['loglevel']}"
    
        os.system(cmd)
        #result = subprocess.run(cmd, capture_output = True)

        tmp_test = cv2.VideoCapture()
        try:
            tmp_test.open(dst_video_uri)
        except:
            return best_qp
        mse = 0.0
        gc = 0.0
        for frame_idx in range(frame_count):

            tmp_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret1, ori_image = tmp_cap.read()
            if ret1 == False:
                continue

            tmp_test.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret2, test_image = tmp_test.read()
            if ret2 == False:
                continue

            mse+=calculate_psnr_ws(ori_image, test_image)

            #dst_video_frame_uri = osp.join(dst_video_folder, 'ori.png')
            #cv2.imwrite(dst_video_frame_uri, ori_image, [cv2.IMWRITE_JPEG_QUALITY, 100])

            #dst_video_frame_uri = osp.join(dst_video_folder, 'test.png')
            #cv2.imwrite(dst_video_frame_uri, test_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
        
        tmp_test.release()
        mse = mse / frame_count
        test_video_size = round((get_video_size(dst_video_uri) + bg_video_size)/ 1000 / 1000 / 1000, 6)
        gc = 1/(alpha * mse + beta * (w_1*test_video_size + 0.000001))

        remove_temp_video(dst_video_uri)
        remove_temp_files(dst_video_folder)

        if (gc > best_gc):
            best_gc = gc
            best_qp = test_qp
        else:
            break
            
    tmp_cap.release()
 
    return best_qp

def segment_video_(ffmpeg_settings, source_video_uri, dst_video_folder, segmentation_info, user_video_spec, user_data, best_qp):
    """
    Segment video tile from the original video

    Parameters
    ----------
    ffmpeg_settings: dict
        ffmpeg related information
    source_video_uri: str
        video uri of original video
    dst_video_folder: str
        folder path of the segmented video tile
    segmentation_info: dict
        tile information
        
    Returns
    -------
        None
    """
    encoding_params = user_data['opt']['e3po_settings']['encoding_params']
    out_w = segmentation_info['segment_out_info']['width']
    out_h = segmentation_info['segment_out_info']['height']
    start_w = segmentation_info['start_position']['width']
    start_h = segmentation_info['start_position']['height']

    chunk_idx = user_video_spec['tile_info']['chunk_idx']
    tile_idx = user_video_spec['tile_info']['tile_idx']
    if tile_idx != -1:  # normal tile stream
        result_video_name = f"chunk_{str(chunk_idx).zfill(4)}_tile_{str(tile_idx).zfill(3)}"
    else:               # background stream
        result_video_name = f"chunk_{str(chunk_idx).zfill(4)}_background"
  

    dst_video_uri = osp.join(dst_video_folder, f'{result_video_name}.mp4')
    #os.chdir(dst_video_folder)
    qp_val = encoding_params['qp_list'][0]
    try: 
        user_data["video_info"]["video_stream"]['color_primaries']
        color_primaries = user_data["video_info"]["video_stream"]['color_primaries']
        color_prim_cmd = 'colorprim='+color_primaries+':transfer='+color_primaries+':colormatrix='+color_primaries
    except:
        color_prim_cmd = ''
   
    cmd = f"{ffmpeg_settings['ffmpeg_path']} " \
          f"-i {source_video_uri} " \
          f"-threads {2} " \
          f"-vf \"crop={out_w}:{out_h}:{start_w}:{start_h}\" " \
          f"-c:v libx264  " \
          f"-preset slower " \
          f" -x264-params \"{color_prim_cmd}\" " \
          f"-r {encoding_params['video_fps']} " \
          f"-qp {best_qp} " \
          f"-y {dst_video_uri} " \
          f"-loglevel {ffmpeg_settings['loglevel']}"
    #settings.logger.debug(cmd)  f"-bf {0} " \ f"-g {32} " \
    #get_logger().debug(cmd)
    os.system(cmd)
    #result = subprocess.run(cmd, capture_output = True)
    return dst_video_uri


def resize_video_(ffmpeg_settings, source_video_uri, dst_video_folder, dst_video_info, user_video_spec, user_data, best_qp):
    """
    Given width and height, resizing the original video.

    Parameters
    ----------
    ffmpeg_settings: dict
        ffmpeg related information
    source_video_uri: str
        video uri of original video
    dst_video_folder: str
        folder path of the segmented video tile
    dst_video_info: dict
        information of the destination video

    Returns
    -------
        None
    """
    encoding_params = user_data['opt']['e3po_settings']['encoding_params']
    chunk_idx = user_video_spec['tile_info']['chunk_idx']
    tile_idx = user_video_spec['tile_info']['tile_idx']
    if tile_idx != -1:  # normal tile stream
        result_video_name = f"chunk_{str(chunk_idx).zfill(4)}_tile_{str(tile_idx).zfill(3)}"
    else:               # background stream
        result_video_name = f"chunk_{str(chunk_idx).zfill(4)}_background"
  

    dst_video_uri = osp.join(dst_video_folder, f'{result_video_name}.mp4')

    dst_video_w = dst_video_info['width']
    dst_video_h = dst_video_info['height']
    
    qp_val = encoding_params['qp_list'][0]
    try: 
        user_data["video_info"]["video_stream"]['color_primaries']
        color_primaries = user_data["video_info"]["video_stream"]['color_primaries']
        color_prim_cmd = 'colorprim='+color_primaries+':transfer='+color_primaries+':colormatrix='+color_primaries
    except:
        color_prim_cmd = ''

    result_frame_path = osp.join(dst_video_folder, f"%d.png")
    cmd = f"{ffmpeg_settings['ffmpeg_path']} " \
          f"-i {source_video_uri} " \
          f"-r {encoding_params['video_fps']} " \
          f"-threads {2} " \
          f"-c:v libx264  " \
          f"-preset slower " \
          f" -x264-params \"{color_prim_cmd}\" " \
          f"-qp {best_qp} " \
          f"-s {dst_video_w}x{dst_video_h} " \
          f"-y {dst_video_uri} " \
          f"-loglevel {ffmpeg_settings['loglevel']}"
    #get_logger().debug(cmd)
    os.system(cmd)
    #result = subprocess.run(cmd, capture_output = True)

    return dst_video_uri


def transcode_video_(source_video_uri, src_proj, dst_proj, src_resolution, dst_resolution, dst_video_folder, chunk_info, ffmpeg_settings, user_data):
    """
    Transcoding videos with different projection formats and different resolutions

    Parameters
    ----------
    source_video_uri: str
        source video uri
    src_proj: str
        source video projection
    dst_proj: str
        destination video projection
    src_resolution: list
        source video resolution with format [height, width]
    dst_resolution: list
        destination video resolution with format [height, width]
    dst_video_folder: str
        path of the destination video
    chunk_info: dict
        chunk information
    ffmpeg_settings: dict
        ffmpeg related information, with format {ffmpeg_path, log_level, thread}

    Returns
    -------
    transcode_video_uri: str
        uri (uniform resource identifier) of the transcode video
    """
    encoding_params = user_data['opt']['e3po_settings']['encoding_params']
    tmp_cap = cv2.VideoCapture()
    assert tmp_cap.open(source_video_uri), f"[error] Can't read video[{source_video_uri}]"
    #temp_source = source_video_uri.split('chunk_0000.mp4')[0] + 'transcode_chunk_0000.mp4'
    #tmp_cap.open(temp_source)

    # Ensure the highest possible quality
    src_video_h = src_resolution[0]
    src_video_w = src_resolution[1]
    dst_video_h = dst_resolution[0]
    dst_video_w = dst_resolution[1]
    dst_res_half = [x//2 for x in dst_resolution]
    src_res_half = [x//2 for x in src_resolution]

    source_video_uri_yuv = source_video_uri.split('mp4')[0] + 'yuv'
    cmd = f"{ffmpeg_settings['ffmpeg_path']} " \
          f"-i {source_video_uri} " \
          f"-r {encoding_params['video_fps']} " \
          f"-pix_fmt yuv420p " \
          f"-s {src_video_w}x{src_video_h} " \
          f"-f rawvideo " \
          f"-y {source_video_uri_yuv} " \
          f"-loglevel {ffmpeg_settings['loglevel']}"
    os.system(cmd)
    
    frame_count = int(tmp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_idx in range(frame_count):
        Y, U, V = yuv_import8(source_video_uri_yuv, src_resolution, num_frames=1, start_frame=frame_idx)
        Y = np.transpose(Y, (1, 2, 0))
        U = np.transpose(U, (1, 2, 0))
        V = np.transpose(V, (1, 2, 0))

        if not Y.any():
            print("Error: Unable to read frame.")
            break
                     
        for channel in range(3):
            if channel == 0:
                pixel_coord = transform_projection_(dst_proj, src_proj, dst_resolution, src_resolution)
                dstMap_u, dstMap_v = cv2.convertMaps(pixel_coord[0].astype(np.float32), pixel_coord[1].astype(np.float32), cv2.CV_16SC2)
                transcode_frame = cv2.remap(Y, dstMap_u, dstMap_v, cv2.INTER_LINEAR)
                transcode_frame_Y = np.reshape(transcode_frame, (1,dst_resolution[0], dst_resolution[1]))

            if channel == 1:
                pixel_coord = transform_projection_(dst_proj, src_proj, dst_res_half, src_res_half)        
                dstMap_u, dstMap_v = cv2.convertMaps(pixel_coord[0].astype(np.float32), pixel_coord[1].astype(np.float32), cv2.CV_16SC2)
                transcode_frame = cv2.remap(U, dstMap_u, dstMap_v, cv2.INTER_LINEAR)
                transcode_frame_U = np.reshape(transcode_frame, (1,dst_res_half[0], dst_res_half[1]))

            if channel == 2:
                pixel_coord = transform_projection_(dst_proj, src_proj, dst_res_half, src_res_half)
                dstMap_u, dstMap_v = cv2.convertMaps(pixel_coord[0].astype(np.float32), pixel_coord[1].astype(np.float32), cv2.CV_16SC2)
                transcode_frame = cv2.remap(V, dstMap_u, dstMap_v, cv2.INTER_LINEAR)
                transcode_frame_V = np.reshape(transcode_frame, (1,dst_res_half[0], dst_res_half[1]))
            
        transcode_frame_uri = osp.join(dst_video_folder, 'transcode_video_uri_yuv.yuv')

        yuv_export(transcode_frame_Y, transcode_frame_U, transcode_frame_V, transcode_frame_uri, dst_resolution, num_frames=1, start_frame=frame_idx, frames=None, yuv444=False)
        #cv2.imwrite(transcode_frame_uri, transcode_frame, [cv2.IMWRITE_JPEG_QUALITY, 100])

    transcode_video_uri = source_video_uri.split("chunk")[0] + 'transcode_chunk_' + str(chunk_info["chunk_idx"]).zfill(4) + '.mp4'
    chunk_idx = chunk_info["chunk_idx"]
    result_video_name = f"chunk_{str(chunk_idx).zfill(4)}_background"
    transcode_video_uri = osp.join(dst_video_folder, f'{result_video_name}.mp4')

    try: 
        user_data["video_info"]["video_stream"]['color_primaries']
        color_primaries = user_data["video_info"]["video_stream"]['color_primaries']
        color_prim_cmd = 'colorprim='+color_primaries+':transfer='+color_primaries+':colormatrix='+color_primaries
    except:
        color_prim_cmd = ''
    # Ensure the highest possible quality
    cmd = f"{ffmpeg_settings['ffmpeg_path']} " \
          f"-f rawvideo " \
          f"-r {encoding_params['video_fps']} " \
          f"-pix_fmt yuv420p " \
          f"-s {dst_video_w}x{dst_video_h} " \
          f"-i {transcode_frame_uri} " \
          f"-c:v libx264 " \
          f"-preset slower " \
          f" -x264-params \"{color_prim_cmd}\" " \
          f"-qp {encoding_params['qp_list'][0]} " \
          f"-y {transcode_video_uri} " \
          f"-loglevel {ffmpeg_settings['loglevel']}"
    os.system(cmd)
    #result = subprocess.run(cmd, capture_output = True)
    remove_temp_files(dst_video_folder)
    remove_temp_video(source_video_uri_yuv)
    remove_temp_video(transcode_frame_uri)

    return transcode_video_uri

def dest_encode_(dst_video_folder, chunk_info, ffmpeg_settings, user_video_spec, json_path, dst_video_uri):
    dst_video_size = get_video_size(dst_video_uri)
    remove_temp_files(dst_video_folder)
    if user_video_spec:
        write_video_json(json_path, dst_video_size, chunk_info, user_video_spec)

def video_analysis(user_data, video_info):
    """
    This API allows users to analyze the full 360 video (if necessary) before the pre-processing starts.
    Parameters
    ----------
    user_data: is initially set to an empy object and users can change it to any structure they need.
    video_info: is a dictionary containing the required video information.

    Returns
    -------
    user_data:
        user should return the modified (or unmodified) user_data as the return value.
        Failing to do so will result in the loss of the information stored in the user_data object.
    """
    
    user_data = user_data or {}
    user_data["video_analysis"] = []#{'Predictions':predictions}
    
    
    return user_data


def init_user(user_data, video_info):
    """
    Initialization function, users initialize their parameters based on the content passed by E3PO

    Parameters
    ----------
    user_data: None
        the initialized user_data is none, where user can store their parameters
    video_info: dict
        video information of original video, user can perform preprocessing according to their requirement

    Returns
    -------
    user_data: dict
        the updated user_data
    """

    user_data = user_data or {}
    user_data["video_info"] = video_info
    user_data["config_params"] = read_config()
    user_data["chunk_idx"] = -1
    
    return user_data


def read_config():
    """
    read the user-customized configuration file as needed

    Returns
    -------
    config_params: dict
        the corresponding config parameters
    """

    config_path = os.path.dirname(os.path.abspath(__file__)) + "/360LCY.yml"
    with open(config_path, 'r', encoding='UTF-8') as f:
        opt = yaml.safe_load(f.read())['approach_settings']

    background_flag = opt['background']['background_flag']
    converted_height = opt['video']['converted']['height']
    converted_width = opt['video']['converted']['width']
    background_height = opt['background']['height']
    background_width = opt['background']['width']
    tile_height_num = opt['video']['tile_height_num']
    tile_width_num = opt['video']['tile_width_num']
    #total_tile_num = tile_height_num * tile_width_num
    total_tile_num = tile_height_num * (tile_width_num-2) + 10
    tile_width = int(opt['video']['converted']['width'] / tile_width_num)
    tile_height = int(opt['video']['converted']['height'] / tile_height_num)
    if background_flag:
        background_info = {
            "width": opt['background']['width'],
            "height": opt['background']['height'],
            "background_projection_mode": opt['background']['projection_mode']
        }
    else:
        background_info = {}

    motion_history_size = opt['video']['hw_size'] * 1000
    motino_prediction_size = opt['video']['pw_size']
    ffmpeg_settings = opt['ffmpeg']
    if not ffmpeg_settings['ffmpeg_path']:
        assert shutil.which('ffmpeg'), '[error] ffmpeg doesn\'t exist'
        ffmpeg_settings['ffmpeg_path'] = shutil.which('ffmpeg')
    else:
        assert os.path.exists(ffmpeg_settings['ffmpeg_path']), \
            f'[error] {ffmpeg_settings["ffmpeg_path"]} doesn\'t exist'
    projection_mode = opt['approach']['projection_mode']
    converted_projection_mode = opt['video']['converted']['projection_mode']

    config_params = {
        "background_flag": background_flag,
        "converted_height": converted_height,
        "converted_width": converted_width,
        "background_height": background_height,
        "background_width": background_width,
        "tile_height_num": tile_height_num,
        "tile_width_num": tile_width_num,
        "total_tile_num": total_tile_num,
        "tile_width": tile_width,
        "tile_height": tile_height,
        "background_info": background_info,
        "motion_history_size": motion_history_size,
        "motion_prediction_size": motino_prediction_size,
        "ffmpeg_settings": ffmpeg_settings,
        "projection_mode": projection_mode,
        "converted_projection_mode": converted_projection_mode
    }

    return config_params


def preprocess_video(source_video_uri, dst_video_folder, chunk_info, user_data, video_info):
    """
    Self defined preprocessing strategy

    Parameters
    ----------
    source_video_uri: str
        the video uri of source video
    dst_video_folder: str
        the folder to store processed video
    chunk_info: dict
        chunk information
    user_data: dict
        store user-related parameters along with their required content
    video_info: dict
        store video information

    Returns
    -------
    user_video_spec: dict
        a dictionary storing user specific information for the preprocessed video
    user_data: dict
        updated user_data
    """
    while True:
        if user_data is None or "video_info" not in user_data:
            user_data = init_user(user_data, video_info)
            user_data["video_info"]["video_stream"] = run_ffprobe(video_info['uri'])
            #if user_data["video_info"]["video_stream"] is None:
            #    user_data["video_info"]["video_stream"]['pix_fmt'] = 'yuv420p'               

            try: user_data["video_info"]["video_stream"]['pix_fmt']
            except: 
                user_data["video_info"]["video_stream"]['pix_fmt'] ='yuv420p'
                user_data["video_info"]["video_stream"]['color_primaries'] = 'bt709'
            

        config_params = user_data['config_params']
        video_info = user_data['video_info']
        user_data["work_folder"] = str(Path(source_video_uri.split('mp4')[0]).parents[0])#dst_video_folder.split('/dst_video_folder')[0]
        opt = get_opt_(user_data["video_info"]["video_stream"], source_video_uri)
        user_data['opt'] = opt

        # update related information
        if user_data['chunk_idx'] == -1:
            user_data['chunk_idx'] = chunk_info['chunk_idx']
            user_data['tile_idx'] = 0
            user_data['transcode_video_uri'] = source_video_uri
        else:
            if user_data['chunk_idx'] != chunk_info['chunk_idx']:
                user_data['chunk_idx'] = chunk_info['chunk_idx']
                user_data['tile_idx'] = 0
                user_data['transcode_video_uri'] = source_video_uri

        # transcoding
        src_projection = config_params['projection_mode']
        dst_projection = config_params['converted_projection_mode']
        
        if user_data['tile_idx'] == 0:
            src_resolution = [video_info['height'],video_info['width']]
            dst_resolution = [config_params['converted_height'], config_params['converted_width']]
            #try:
            #    best_qp = get_rd_analysis(config_params['ffmpeg_settings'], source_video_uri, dst_video_folder, user_data, chunk_info)
            #except:
                
            best_qp = user_data['opt']['e3po_settings']['encoding_params']['qp_list'][0]                        
            selected_tiles = viewport_prediction_chunk(source_video_uri, src_projection, dst_projection, src_resolution, dst_resolution, dst_video_folder, chunk_info, config_params, user_data)                       

        if src_projection != dst_projection and user_data['tile_idx'] == 0:

            user_data['transcode_video_uri'] = transcode_video_(
                source_video_uri, src_projection, dst_projection, src_resolution, dst_resolution,
                dst_video_folder, chunk_info, config_params['ffmpeg_settings'], user_data
            )
        else:
            pass
        transcode_video_uri = user_data['transcode_video_uri']

        # segmentation
        if user_data['tile_idx'] < config_params['total_tile_num']:
            if user_data['tile_idx'] in selected_tiles: 
                tile_info, segment_info = tile_segment_info(chunk_info, user_data)
                #segment_video(config_params['ffmpeg_settings'], transcode_video_uri, dst_video_folder, segment_info)
                user_video_spec = {'segment_info': segment_info, 'tile_info': tile_info}
                dst_video_uri = segment_video_(config_params['ffmpeg_settings'], transcode_video_uri, dst_video_folder, segment_info, user_video_spec, user_data, best_qp)
            else:
                user_video_spec = None
            user_data['tile_idx'] += 1
            
        # resize, background stream 
        # user_data['tile_idx'] != len(selected_tiles) and 
        elif config_params['background_flag'] and user_data['tile_idx'] == config_params['total_tile_num']:
            #elif user_data['tile_idx'] == config_params['total_tile_num'] and config_params['background_flag']:
            bg_projection = config_params['background_info']['background_projection_mode']
            if bg_projection == src_projection:
                bg_video_uri = source_video_uri
            else:
                src_resolution = [video_info['height'], video_info['width']]
                bg_resolution = [config_params['background_height'], config_params['background_width']]
                

                if user_data["video_info"]["video_stream"]['pix_fmt'] == 'yuv420p':
                    bg_video_uri = transcode_video_(
                    source_video_uri, src_projection, bg_projection, src_resolution, bg_resolution,
                    dst_video_folder, chunk_info, config_params['ffmpeg_settings'], user_data
                )
                else:
                    bg_video_uri = transcode_video(
                source_video_uri, src_projection, bg_projection, src_resolution, bg_resolution,
                dst_video_folder, chunk_info, config_params['ffmpeg_settings']
            )                

            #resize_video(config_params['ffmpeg_settings'], bg_video_uri, dst_video_folder, config_params['background_info'])

            user_data['tile_idx'] += 1
            #if ((user_data['tile_idx'] - 1) != len(selected_tiles)):
            #   config_params['background_info']['width'] = config_params['background_width']*2 
            #   config_params['background_info']['height'] = config_params['background_height']*2

            user_video_spec = {
                'segment_info': config_params['background_info'],
                'tile_info': {'chunk_idx': chunk_info['chunk_idx'], 'tile_idx': -1}
            }

            dst_video_uri = resize_video_(config_params['ffmpeg_settings'], bg_video_uri, dst_video_folder, config_params['background_info'], user_video_spec, user_data, best_qp)
           
        else:
            user_video_spec = None
            break

        json_path = osp.join(user_data["work_folder"], 'video_size.json')
        #dest_encode(dst_video_folder, chunk_info, config_params['ffmpeg_settings'], user_video_spec, json_path)
        if user_video_spec:
            dest_encode_(dst_video_folder, chunk_info, config_params['ffmpeg_settings'], user_video_spec, json_path, dst_video_uri)
    
    curr_path = os.path.dirname(os.path.abspath(__file__)) 
    tile_dir_json_path=os.path.join(curr_path,"tile_dir.json")
    write_tile_dir_json(tile_dir_json_path, chunk_info['chunk_idx'], dst_video_folder)

    return user_video_spec, user_data


def download_decision(network_stats, motion_history, video_size, curr_ts, user_data, video_info):
    """
    Self defined download strategy

    Parameters
    ----------
    network_stats: list
        a list represents historical network status
    motion_history: list
        a list represents historical motion information
    video_size: dict
        video size of preprocessed video
    curr_ts: int
        current system timestamp
    user_data: dict
        user related parameters and information
    video_info: dict
        video information for decision module

    Returns
    -------
    dl_list: list
        the list of tiles which are decided to be downloaded
    user_data: dict
        updated user_date
    """
    try:
        if user_data is None or "video_info" not in user_data:
            user_data = init_user(user_data, video_info)
            user_data['last_img_index'] = 0
            curr_path = os.path.dirname(os.path.abspath(__file__)) 
            json_path=os.path.join(curr_path,"tile_dir.json")
            result_chunk_name = f"chunk_{str(0)}"
            tile_dir_ =  read_tile_dir_json(json_path)
            tile_dir = tile_dir_[result_chunk_name]['dst_folder']
            if "\\" in tile_dir:
                test_group_ = tile_dir.split('\\')[-4]
                video_name_ = tile_dir.split('\\')[-3]
                test_approach_name = tile_dir.split('\\')[-2]
            elif '/' in tile_dir:
                test_group_ = tile_dir.split('/')[-4]
                video_name_ = tile_dir.split('/')[-3]
                test_approach_name = tile_dir.split('/')[-2]
            else:
                test_group_ = ''
                video_name_ = ''
                test_approach_name = ''

            project_dir = os.path.dirname(os.path.abspath(__file__)).split('approaches')[0]
            project_path = project_dir[:-1]
            parser = argparse.ArgumentParser()
            parser.add_argument('-approach_name', type=str, required=True,
                            help="test approach name")
            parser.add_argument('-approach_type', type=str, required=True,
                            help="approach type")
            args = parser.parse_args()
            approach_name = args.approach_name

            user_data['decision_json_uri'] = os.path.join(
                project_path,
                'result',
                test_group_,
                #str(Path(video_info['uri']),
                video_name_,
                approach_name,
                'decision.json'
            )

            user_data['code_branch'] = False

        if curr_ts == 0:  # initialize the related parameters
                user_data['next_download_idx'] = 0
                user_data['latest_decision'] = []
                user_data['curr_ts_update'] = {'chunk_idx':0, 'curr_ts':curr_ts}
                #user_data['video_analysis'].append({'chunk_idx':data['next_download_idx'], 'tile_id': "", 'data':None, 'update':False} )

        if (os.path.exists(user_data['decision_json_uri'])):
            user_data['code_branch'] = True
        else:
            user_data['code_branch'] = False

        if (user_data['code_branch']):
            config_params = user_data['config_params']
            video_info = user_data['video_info']

            
            dl_list = []
            chunk_idx = user_data['next_download_idx']
            latest_decision = user_data['latest_decision']

            if (user_data['curr_ts_update']['chunk_idx'] < chunk_idx):
                    user_data['curr_ts_update'] = {'chunk_idx':chunk_idx, 'curr_ts':curr_ts}

            if curr_ts == 0:
                frame_idx =0
            else:
                frame_idx = int((curr_ts - video_info['pre_download_duration']) * 30 // 1000.0)
    
            if user_data['last_img_index'] == frame_idx and curr_ts!= 0:
                return dl_list, user_data
            user_data['last_img_index'] = frame_idx

            if user_data['next_download_idx'] >= video_info['duration'] / video_info['chunk_duration']:
                return dl_list, user_data

            predicted_record = predict_motion_tile_(motion_history, config_params['motion_history_size'], config_params['motion_prediction_size'])  # motion prediction
            tile_record = tile_decision_(predicted_record, video_size, video_info['range_fov'], chunk_idx, user_data)     # tile decision
        
            #dl_list = generate_dl_list(chunk_idx, tile_record, latest_decision, dl_list)

            user_data, dl_list = update_decision_info(user_data, tile_record, curr_ts, dl_list)            # update decision information

            #if (user_data['video_analysis'][chunk_idx]['chunk_idx'] == chunk_idx):
            #    dl_list = []

            return dl_list, user_data
    
        else:
            dl_list, user_data = download_decision2(network_stats, motion_history, video_size, curr_ts, user_data, video_info)
            return dl_list, user_data
   
    except:
        dl_list, user_data = download_decision2(network_stats, motion_history, video_size, curr_ts, user_data, video_info)
        return dl_list, user_data


def generate_display_result(curr_display_frames, current_display_chunks, curr_fov, dst_video_frame_uri, frame_idx, video_size, user_data, video_info):
    """
    Generate fov images corresponding to different approaches

    Parameters
    ----------
    curr_display_frames: list
        current available video tile frames
    current_display_chunks: list
        current available video chunks
    curr_fov: dict
        current fov information, with format {"curr_motion", "range_fov", "fov_resolution"}
    dst_video_frame_uri: str
        the uri of generated fov frame
    frame_idx: int
        frame index of current display frame
    video_size: dict
        video size of preprocessed video
    user_data: dict
        user related parameters and information
    video_info: dict
        video information for evaluation

    Returns
    -------
    user_data: dict
        updated user_data
    """

    get_logger().debug(f'[evaluation] start get display img {frame_idx}')

    if user_data is None or "video_info" not in user_data:
        user_data = init_user(user_data, video_info)

    video_info = user_data['video_info']
    config_params = user_data['config_params']

    chunk_idx = int(frame_idx * (1000 / video_info['video_fps']) // (video_info['chunk_duration'] * 1000))  # frame idx starts from 0
    if chunk_idx <= len(current_display_chunks) - 1:
        tile_list = current_display_chunks[chunk_idx]['tile_list']
    else:
        tile_list = current_display_chunks[-1]['tile_list']

    avail_tile_list = []
    for i in range(len(tile_list)):
        tile_id = tile_list[i]['tile_id']
        tile_idx = video_size[tile_id]['user_video_spec']['tile_info']['tile_idx']
        avail_tile_list.append(tile_idx)

    # calculating fov_uv parameters
    fov_ypr = [float(curr_fov['curr_motion']['yaw']), float(curr_fov['curr_motion']['pitch']), 0]
    _3d_polar_coord = fov_to_3d_polar_coord(fov_ypr, curr_fov['range_fov'], curr_fov['fov_resolution'])
    pixel_coord = _3d_polar_coord_to_pixel_coord(_3d_polar_coord, config_params['projection_mode'], [config_params['converted_height'], config_params['converted_width']])

    coord_tile_list = pixel_coord_to_tile_corrected(pixel_coord, config_params['total_tile_num'], video_size, chunk_idx, avail_tile_list)
    relative_tile_coord = pixel_coord_to_relative_tile_coord_corrected(pixel_coord, coord_tile_list, video_size, chunk_idx)
    
    unique_tile_list = np.unique(coord_tile_list)
    unavail_pixel_coord = ~np.isin(coord_tile_list, avail_tile_list)    # calculate the pixels that have not been transmitted.
    coord_tile_list[unavail_pixel_coord] = -1

    display_img = np.full((coord_tile_list.shape[0], coord_tile_list.shape[1], 3), [128, 128, 128], dtype=np.float32)  # create an empty matrix for the final image

    for i, tile_idx in enumerate(avail_tile_list):
        hit_coord_mask = (coord_tile_list == tile_idx)
        if not np.any(hit_coord_mask):  # if no pixels belong to the current frame, skip
            continue

        if tile_idx != -1:
            dstMap_u, dstMap_v = cv2.convertMaps(relative_tile_coord[0].astype(np.float32), relative_tile_coord[1].astype(np.float32), cv2.CV_16SC2)
        else:
            result_video_name = f"chunk_{str(chunk_idx).zfill(4)}_background"
            #config_params['background_info']['width'] = video_size[result_video_name]['user_video_spec']['segment_info']['width']
            #config_params['background_info']['height'] = video_size[result_video_name]['user_video_spec']['segment_info']['height']
            out_pixel_coord = _3d_polar_coord_to_pixel_coord(
                _3d_polar_coord,
                config_params['background_info']['background_projection_mode'],
                [2*config_params['background_info']['height'], 2*config_params['background_info']['width']]
            )
            dstMap_u, dstMap_v = cv2.convertMaps(out_pixel_coord[0].astype(np.float32), out_pixel_coord[1].astype(np.float32), cv2.CV_16SC2)
        if tile_idx != -1:
            remapped_frame = cv2.remap(curr_display_frames[i], dstMap_u, dstMap_v, 1)
        else:
            dim = (2*config_params['background_info']['width'], 2*config_params['background_info']['height'])
            resized = cv2.resize(curr_display_frames[i], dim, interpolation = cv2.INTER_LANCZOS4)
            remapped_frame = cv2.remap(resized, dstMap_u, dstMap_v, 1)
        display_img[hit_coord_mask] = remapped_frame[hit_coord_mask]

    cv2.imwrite(dst_video_frame_uri, display_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

    get_logger().debug(f'[evaluation] end get display img {frame_idx}')

    return user_data


def update_decision_info(user_data, tile_record, curr_ts, dl_list):
    """
    update the decision information

    Parameters
    ----------
    user_data: dict
        user related parameters and information
    tile_record: list
        recode the tiles should be downloaded
    curr_ts: int
        current system timestamp

    Returns
    -------
    user_data: dict
        updated user_data
    """

    # update latest_decision
    for i in range(len(tile_record)):
        if tile_record[i] not in user_data['latest_decision']:
            user_data['latest_decision'].append(tile_record[i])
    #if user_data['config_params']['background_flag']:
    #    if -1 not in user_data['latest_decision']:
    #        user_data['latest_decision'].append(-1)

    # update chunk_idx & latest_decision
    if curr_ts == 0 or curr_ts >= user_data['video_info']['pre_download_duration'] \
            + user_data['next_download_idx'] * user_data['video_info']['chunk_duration'] * 1000:

        if (-1 not in user_data['latest_decision'] and len(user_data['latest_decision']) !=(user_data['config_params']['total_tile_num'])):
            if ( len(user_data['latest_decision']) == (user_data['config_params']['total_tile_num'] - 1)):
                for i in range(user_data['config_params']['total_tile_num']):
                    if i not in tile_record:
                        tile_record.append(i)
            else:
                tile_record.append(-1)
               
        tile_record_x = [x for x in user_data['latest_decision']]
        
        if (all(x in tile_record_x for x in [1, 2, 3, 4, 5, 6, 7, 8])):
            tile_record_x = []
            tile_record_x.append(16)

        elif (all(x in tile_record_x for x in [2, 3, 6, 7])):
            tile_record_x.remove(2)
            tile_record_x.remove(3)
            tile_record_x.remove(6)
            tile_record_x.remove(7)
            tile_record_x.append(10)
            
            if (all(x in tile_record_x for x in [1, 5])):
                tile_record_x.remove(1)
                tile_record_x.remove(5)
                tile_record_x.append(12)
            
            if (all(x in tile_record_x for x in [4, 8])):
                tile_record_x.remove(4)
                tile_record_x.remove(8)
                tile_record_x.append(15)

        elif (all(x in tile_record_x for x in [1, 2, 5, 6])):
            tile_record_x.remove(1)
            tile_record_x.remove(2)
            tile_record_x.remove(5)
            tile_record_x.remove(6)
            tile_record_x.append(9)
            
            if (all(x in tile_record_x for x in [3, 7])):
                tile_record_x.remove(3)
                tile_record_x.remove(7)
                tile_record_x.append(14)
            
            if (all(x in tile_record_x for x in [4, 8])):
                tile_record_x.remove(4)
                tile_record_x.remove(8)
                tile_record_x.append(15)

        elif (all(x in tile_record_x for x in [3, 4, 7, 8])):
            tile_record_x.remove(3)
            tile_record_x.remove(4)
            tile_record_x.remove(7)
            tile_record_x.remove(8)
            tile_record_x.append(11)
            
            if (all(x in tile_record_x for x in [1, 5])):
                tile_record_x.remove(1)
                tile_record_x.remove(5)
                tile_record_x.append(12)
            
            if (all(x in tile_record_x for x in [2, 6])):
                tile_record_x.remove(2)
                tile_record_x.remove(6)
                tile_record_x.append(13)

        if (all(x in tile_record_x for x in [1, 5])):
            tile_record_x.remove(1)
            tile_record_x.remove(5)
            tile_record_x.append(12)
        if (all(x in tile_record_x for x in [2, 6])):
            tile_record_x.remove(2)
            tile_record_x.remove(6)
            tile_record_x.append(13)
        if (all(x in tile_record_x for x in [3, 7])):
            tile_record_x.remove(3)
            tile_record_x.remove(7)
            tile_record_x.append(14)
        if (all(x in tile_record_x for x in [4, 8])):
            tile_record_x.remove(4)
            tile_record_x.remove(8)
            tile_record_x.append(15)
        #if (9 in user_data['latest_decision']):
        #    if 2 in final_tile_set:
        #        final_tile_set.remove(2)
        #    if 3 in final_tile_set:
        #        final_tile_set.remove(3)
        #    if 6 in final_tile_set:
        #        final_tile_set.remove(6)
        #    if 7 in final_tile_set:
        #        final_tile_set.remove(7)

        #if (10 in user_data['latest_decision']):
        #    final_tile_set = []
        tile_record_x.append(-1)
        user_data['latest_decision'] = []
        dl_list = generate_dl_list(user_data['next_download_idx'], tile_record_x, user_data['latest_decision'], dl_list)
        write_decision_json(user_data['decision_json_uri'], user_data['curr_ts_update']['curr_ts'], dl_list)
        dl_list = []

        user_data['next_download_idx'] += 1
        user_data['latest_decision'] = []

    return user_data, dl_list

def tile_decision_(predicted_record, video_size, range_fov, chunk_idx, user_data):
    """
    Deciding which tiles should be transmitted for each chunk, within the prediction window
    (As an example, users can implement their customized function.)

    Parameters
    ----------
    predicted_record: dict
        the predicted motion, with format {yaw: , pitch: , scale:}, where
        the parameter 'scale' is used for transcoding approach
    video_size: dict
        the recorded whole video size after video preprocessing
    range_fov: list
        degree range of fov, with format [height, width]
    chunk_idx: int
        index of current chunk
    user_data: dict
        user related data structure, necessary information for tile decision

    Returns
    -------
    tile_record: list
        the decided tile list of current update, each item is the chunk index
    """
    # The current tile decision method is to sample the fov range corresponding to the predicted motion of each chunk,
    # and the union of the tile sets mapped by these sampling points is the tile set to be transmitted.

    #if (user_data['video_analysis'][chunk_idx]['data'] == None):
    #    user_data['video_analysis'][chunk_idx]['background_tile_id'] = f"chunk_{str(chunk_idx).zfill(4)}_background"
    #    tile_idx = config_params['total_tile_num']
    #    tile_id = f"chunk_{str(chunk_idx).zfill(4)}_tile_{str(tile_idx).zfill(3)}"
    #    user_data['video_analysis'][chunk_idx]['tile_id'] = tile_id
    #    predicted_point = predicted_record[0]
    #    _3d_polar_coord = fov_to_3d_polar_coord([float(predicted_motion['yaw']), float(predicted_motion['pitch']), 0], range_fov, [600, 600])
    #    pixel_coord = pixel_coord_to_tile_corrected(_3d_polar_coord, config_params['projection_mode'], [converted_height, converted_width])
        

    curr_path = os.path.dirname(os.path.abspath(__file__)) 
    json_path=os.path.join(curr_path,"tile_prediction.json")
    result_chunk_name = f"chunk_{str(chunk_idx)}"
    tile_prediction =  read_tile_prediction_json(json_path)
    
    config_params = user_data['config_params']
    tile_record = []
    sampling_size = [50, 50]
    converted_width = user_data['config_params']['converted_width']
    converted_height = user_data['config_params']['converted_height']
    for predicted_motion in predicted_record:
        _3d_polar_coord = fov_to_3d_polar_coord([float(predicted_motion['yaw']), float(predicted_motion['pitch']), 0], range_fov, sampling_size)
        pixel_coord = _3d_polar_coord_to_pixel_coord(_3d_polar_coord, config_params['projection_mode'], [converted_height, converted_width])
        coord_tile_list = pixel_coord_to_tile_corrected_decision_making(pixel_coord, config_params['total_tile_num'], video_size, chunk_idx)
        unique_tile_list = [int(item) for item in np.unique(coord_tile_list)]
        tile_record.extend(unique_tile_list)

    if 0 in tile_record:
        tile_record.remove(0)
    if (config_params['total_tile_num']-1) in tile_record:
        tile_record.remove(config_params['total_tile_num']-1)

    final_tile_set = [i for i in tile_record if i in tile_prediction[result_chunk_name]['Predicted_tiles']]

    #if (all(x in final_tile_set for x in [2, 3, 6, 7])):
    #    if(all(x not in user_data['latest_decision']) for x in [2, 3, 6, 7]):
    #        final_tile_set.remove(2)
    #        final_tile_set.remove(3)
    #        final_tile_set.remove(6)
    #        final_tile_set.remove(7)
    #        final_tile_set.append(9)


    #if (9 in user_data['latest_decision']):
    #    if 2 in final_tile_set:
    #        final_tile_set.remove(2)
    #    if 3 in final_tile_set:
    #        final_tile_set.remove(3)
    #    if 6 in final_tile_set:
    #        final_tile_set.remove(6)
    #    if 7 in final_tile_set:
    #        final_tile_set.remove(7)

    #if (10 in user_data['latest_decision']):
    #    final_tile_set = []
    
    #if config_params['background_flag']:
    #    if (-1 not in user_data['latest_decision']):
    #        final_tile_set.append(-1)

    return final_tile_set

def predict_motion_tile_(motion_history, motion_history_size, motion_prediction_size):
    """
    Predicting motion with given historical information and prediction window size.
    (As an example, users can implement their customized function.)

    Parameters
    ----------
    motion_history: dict
        a dictionary recording the historical motion, with the following format:

    motion_history_size: int
        the size of motion history to be used for predicting
    motion_prediction_size: int
        the size of motion to be predicted

    Returns
    -------
    list
        The predicted record list, which sequentially store the predicted motion of the future pw chunks.
         Each motion dictionary is stored in the following format:
            {'yaw ': yaw,' pitch ': pitch,' scale ': scale}
    """
    # Use exponential smoothing to predict the angle of each motion within pw for yaw and pitch.
    a = 0.3  # Parameters for exponential smoothing prediction
    hw = [d['motion_record'] for d in motion_history]
    predicted_motion = list(hw)[-1]
    #for motion_record in list(hw)[-motion_history_size:]:
    #    predicted_motion['yaw'] = a * predicted_motion['yaw'] + (1-a) * motion_record['yaw']
    #    predicted_motion['pitch'] = a * predicted_motion['pitch'] + (1-a) * motion_record['pitch']
    #    predicted_motion['scale'] = a * predicted_motion['scale'] + (1-a) * motion_record['scale']

    # The current prediction method implemented is to use the same predicted motion for all chunks in pw.
    predicted_record = []
    for i in range(motion_prediction_size):
        predicted_record.append(deepcopy(predicted_motion))

    return predicted_record

def tile_segment_info(chunk_info, user_data):
    """
    Generate the information for the current tile, with required format
    Parameters
    ----------
    chunk_info: dict
        chunk information
    user_data: dict
        user related information

    Returns
    -------
    tile_info: dict
        tile related information, with format {chunk_idx:, tile_idx:}
    segment_info: dict
        segmentation related information, with format
        {segment_out_info:{width:, height:}, start_position:{width:, height:}}
    """

    tile_idx = user_data['tile_idx']

    config_params = user_data['config_params']
    dst_projection = config_params['converted_projection_mode']
        
    dst_height = config_params['converted_height']
    dst_width = config_params['converted_width']
    encoding_params = user_data['opt']['e3po_settings']['encoding_params']
    tile_idx = user_data['tile_idx']
    
    if (tile_idx == 0):
        index_width = 0
        index_height = 0
        tile_width = dst_width
        tile_height = int(dst_height/6)
        qp = encoding_params['qp_list'][0]
    elif (tile_idx == (config_params['total_tile_num']-1)):
        index_width = 0
        index_height = dst_height - int(dst_height/6)
        tile_width = dst_width
        tile_height = int(dst_height/6)
        qp = encoding_params['qp_list'][0]
    elif (tile_idx == (9)):
        index_width =  0
        index_height = int(dst_height/6)
        tile_width = int(dst_width/4*2)
        tile_height = int(dst_height/3*2)
        qp = encoding_params['qp_list'][0]
    elif (tile_idx == (10)):
        index_width =  int(dst_width/4)
        index_height = int(dst_height/6)
        tile_width = int(dst_width/4*2)
        tile_height = int(dst_height/3*2)
        qp = encoding_params['qp_list'][0]
    elif (tile_idx == (11)):
        index_width =  int(dst_width/4*2)
        index_height = int(dst_height/6)
        tile_width = int(dst_width/4*2)
        tile_height = int(dst_height/3*2)
        qp = encoding_params['qp_list'][0]
    elif (tile_idx == (12)):
        index_width =  0
        index_height = int(dst_height/6)
        tile_width = int(dst_width/4)
        tile_height = int(dst_height/3*2)
        qp = encoding_params['qp_list'][0]
    elif (tile_idx == (13)):
        index_width =  int(dst_width/4)
        index_height = int(dst_height/6)
        tile_width = int(dst_width/4)
        tile_height = int(dst_height/3*2)
        qp = encoding_params['qp_list'][0]
    elif (tile_idx == (14)):
        index_width =  int(dst_width/4*2)
        index_height = int(dst_height/6)
        tile_width = int(dst_width/4)
        tile_height = int(dst_height/3*2)
        qp = encoding_params['qp_list'][0]
    elif (tile_idx == (15)):
        index_width =  int(dst_width/4*3)
        index_height = int(dst_height/6)
        tile_width = int(dst_width/4)
        tile_height = int(dst_height/3*2)
        qp = encoding_params['qp_list'][0]
    elif (tile_idx == (config_params['total_tile_num']-2)):
        index_width =  0
        index_height = int(dst_height/6)
        tile_width = int(dst_width/4*4)
        tile_height = int(dst_height/3*2)
        qp = encoding_params['qp_list'][0]
    else:
        index_width = ((tile_idx-1) % user_data['config_params']['tile_width_num'])* int(dst_width/4)
        index_height = (tile_idx-1) // user_data['config_params']['tile_width_num'] * int(dst_height/3) + int(dst_height/6)
        tile_width = int(dst_width/4)
        tile_height = int(dst_height/3)
        qp = encoding_params['qp_list'][0]
        
        
    segment_info = {
        'segment_out_info': {
            'width': tile_width,
            'height': tile_height
        },
        'start_position': {
            'width': index_width ,
            'height': index_height
        },
        'qp': qp
    }

    tile_info = {
        'chunk_idx': user_data['chunk_idx'],
        'tile_idx': tile_idx
        }
    
    
    #index_width = tile_idx % user_data['config_params']['tile_width_num']        # determine which col
    #index_height = tile_idx // user_data['config_params']['tile_width_num']      # determine which row

    #segment_info = {
    #    'segment_out_info': {
    #        'width': user_data['config_params']['tile_width'],
    #        'height': user_data['config_params']['tile_height']
    #    },
    #    'start_position': {
    #        'width': index_width * user_data['config_params']['tile_width'],
    #        'height': index_height * user_data['config_params']['tile_height']
    #    }
    #}

    #tile_info = {
    #    'chunk_idx': user_data['chunk_idx'],
    #    'tile_idx': tile_idx
    #}

    return tile_info, segment_info


def write_tile_prediction_json(json_path, chunk_idx, Predicted_tiles):
    
    fpath, _ = os.path.split(json_path)
    os.makedirs(fpath, exist_ok=True)

    result_video_name = f"chunk_{str(chunk_idx)}"

    if os.path.exists(json_path):
        with open(json_path, 'r') as file:
            json_data = json.load(file)
        json_data[result_video_name] = {
            'Chunk_index': chunk_idx,
            'Predicted_tiles': Predicted_tiles            
        }
        with open(json_path, 'w') as file:
            json.dump(json_data, file, indent=2, sort_keys=True)
    else:
        with open(json_path, "w", encoding='utf-8') as file:
            json_data = {
                result_video_name: {
                    'Chunk_index': chunk_idx,
                    'Predicted_tiles': Predicted_tiles   
                }
            }
            json.dump(json_data, file, indent=2, sort_keys=True)

def read_tile_prediction_json(json_path):
   
    try:
        with open(json_path, encoding='UTF-8') as f:
            video_json = json.load(f)
        return video_json
    except Exception as e:
        raise ValueError(f"Error reading file: {json_path}")


def write_tile_dir_json(json_path, chunk_idx, dst_folder):
    
    fpath, _ = os.path.split(json_path)
    os.makedirs(fpath, exist_ok=True)

    result_video_name = f"chunk_{str(chunk_idx)}"

    if os.path.exists(json_path):
        with open(json_path, 'r') as file:
            json_data = json.load(file)
        json_data[result_video_name] = {
            'Chunk_index': chunk_idx,
            'dst_folder': dst_folder            
        }
        with open(json_path, 'w') as file:
            json.dump(json_data, file, indent=2, sort_keys=True)
    else:
        with open(json_path, "w", encoding='utf-8') as file:
            json_data = {
                result_video_name: {
                    'Chunk_index': chunk_idx,
                    'dst_folder': dst_folder   
                }
            }
            json.dump(json_data, file, indent=2, sort_keys=True)

def read_tile_dir_json(json_path):
   
    try:
        with open(json_path, encoding='UTF-8') as f:
            video_json = json.load(f)
        return video_json
    except Exception as e:
        raise ValueError(f"Error reading file: {json_path}")


def download_decision2(network_stats, motion_history, video_size, curr_ts, user_data, video_info):
    """
    Self defined download strategy

    Parameters
    ----------
    network_stats: list
        a list represents historical network status
    motion_history: list
        a list represents historical motion information
    video_size: dict
        video size of preprocessed video
    curr_ts: int
        current system timestamp
    user_data: dict
        user related parameters and information
    video_info: dict
        video information for decision module

    Returns
    -------
    dl_list: list
        the list of tiles which are decided to be downloaded
    user_data: dict
        updated user_date
    """

    if user_data is None or "video_info" not in user_data:
        user_data = init_user(user_data, video_info)

    config_params = user_data['config_params']
    video_info = user_data['video_info']

    if curr_ts == 0:  # initialize the related parameters
        user_data['next_download_idx'] = 0
        user_data['latest_decision'] = []
    dl_list = []
    chunk_idx = user_data['next_download_idx']
    latest_decision = user_data['latest_decision']

    if user_data['next_download_idx'] >= video_info['duration'] / video_info['chunk_duration']:
        return dl_list, user_data

    predicted_record = predict_motion_tile_(motion_history, config_params['motion_history_size'], config_params['motion_prediction_size'])  # motion prediction
    try:
        tile_record = tile_decision2(predicted_record, video_size, video_info['range_fov'], chunk_idx, user_data)     # tile decision
    except:
        tile_record = tile_decision3(predicted_record, video_size, video_info['range_fov'], chunk_idx, user_data)     # tile decision
    dl_list = generate_dl_list(chunk_idx, tile_record, latest_decision, dl_list)

    user_data = update_decision_info2(user_data, tile_record, curr_ts)            # update decision information

    return dl_list, user_data


def update_decision_info2(user_data, tile_record, curr_ts):
    """
    update the decision information

    Parameters
    ----------
    user_data: dict
        user related parameters and information
    tile_record: list
        recode the tiles should be downloaded
    curr_ts: int
        current system timestamp

    Returns
    -------
    user_data: dict
        updated user_data
    """

    # update latest_decision
    for i in range(len(tile_record)):
        if tile_record[i] not in user_data['latest_decision']:
            user_data['latest_decision'].append(tile_record[i])
    if user_data['config_params']['background_flag']:
        if -1 not in user_data['latest_decision']:
            user_data['latest_decision'].append(-1)

    # update chunk_idx & latest_decision
    if curr_ts == 0 or curr_ts >= user_data['video_info']['pre_download_duration'] \
            + user_data['next_download_idx'] * user_data['video_info']['chunk_duration'] * 1000:
        user_data['next_download_idx'] += 1
        user_data['latest_decision'] = []

    return user_data

def tile_decision2(predicted_record, video_size, range_fov, chunk_idx, user_data):
    """
    Deciding which tiles should be transmitted for each chunk, within the prediction window
    (As an example, users can implement their customized function.)

    Parameters
    ----------
    predicted_record: dict
        the predicted motion, with format {yaw: , pitch: , scale:}, where
        the parameter 'scale' is used for transcoding approach
    video_size: dict
        the recorded whole video size after video preprocessing
    range_fov: list
        degree range of fov, with format [height, width]
    chunk_idx: int
        index of current chunk
    user_data: dict
        user related data structure, necessary information for tile decision

    Returns
    -------
    tile_record: list
        the decided tile list of current update, each item is the chunk index
    """
    # The current tile decision method is to sample the fov range corresponding to the predicted motion of each chunk,
    # and the union of the tile sets mapped by these sampling points is the tile set to be transmitted.

    #if (user_data['video_analysis'][chunk_idx]['data'] == None):
    #    user_data['video_analysis'][chunk_idx]['background_tile_id'] = f"chunk_{str(chunk_idx).zfill(4)}_background"
    #    tile_idx = config_params['total_tile_num']
    #    tile_id = f"chunk_{str(chunk_idx).zfill(4)}_tile_{str(tile_idx).zfill(3)}"
    #    user_data['video_analysis'][chunk_idx]['tile_id'] = tile_id
    #    predicted_point = predicted_record[0]
    #    _3d_polar_coord = fov_to_3d_polar_coord([float(predicted_motion['yaw']), float(predicted_motion['pitch']), 0], range_fov, [600, 600])
    #    pixel_coord = pixel_coord_to_tile_corrected(_3d_polar_coord, config_params['projection_mode'], [converted_height, converted_width])
        
    try:
        curr_path = os.path.dirname(os.path.abspath(__file__)) 
        json_path=os.path.join(curr_path,"tile_prediction.json")
        result_chunk_name = f"chunk_{str(chunk_idx)}"
        tile_prediction =  read_tile_prediction_json(json_path)
    except:
        tile_prediction = [1,2,3,4,5,6,7,8]
    
    config_params = user_data['config_params']
    tile_record = []
    sampling_size = [50, 50]
    converted_width = user_data['config_params']['converted_width']
    converted_height = user_data['config_params']['converted_height']
    for predicted_motion in predicted_record:
        _3d_polar_coord = fov_to_3d_polar_coord([float(predicted_motion['yaw']), float(predicted_motion['pitch']), 0], range_fov, sampling_size)
        pixel_coord = _3d_polar_coord_to_pixel_coord(_3d_polar_coord, config_params['projection_mode'], [converted_height, converted_width])
        coord_tile_list = pixel_coord_to_tile_corrected_decision_making(pixel_coord, config_params['total_tile_num'], video_size, chunk_idx)
        unique_tile_list = [int(item) for item in np.unique(coord_tile_list)]
        tile_record.extend(unique_tile_list)

    if 0 in tile_record:
        tile_record.remove(0)
    if (config_params['total_tile_num']-1) in tile_record:
        tile_record.remove(config_params['total_tile_num']-1)

    final_tile_set = [i for i in tile_record if i in tile_prediction[result_chunk_name]['Predicted_tiles']]

    tile_record_x = [x for x in final_tile_set]
    
    
    if (all(x in tile_record_x for x in [1, 2, 3, 4, 5, 6, 7, 8])):
        if 16 not in user_data['latest_decision']:
            tile_record_x = []
            tile_record_x.append(16)
    if 16 not in user_data['latest_decision']:

        if (9 in user_data['latest_decision']):
            if 1 in tile_record_x:
                tile_record_x.remove(1)
            if 2 in tile_record_x:
                tile_record_x.remove(2)
            if 5 in tile_record_x:
                tile_record_x.remove(5)
            if 6 in tile_record_x:
                tile_record_x.remove(6)
            if 12 in tile_record_x:
                tile_record_x.remove(12)
            if 13 in tile_record_x:
                tile_record_x.remove(13)

        if (10 in user_data['latest_decision']):
            if 2 in tile_record_x:
                tile_record_x.remove(2)
            if 3 in tile_record_x:
                tile_record_x.remove(3)
            if 6 in tile_record_x:
                tile_record_x.remove(6)
            if 7 in tile_record_x:
                tile_record_x.remove(7)
            if 13 in tile_record_x:
                tile_record_x.remove(13)
            if 14 in tile_record_x:
                tile_record_x.remove(14)

        if (11 in user_data['latest_decision']):
            if 3 in tile_record_x:
                tile_record_x.remove(3)
            if 4 in tile_record_x:
                tile_record_x.remove(4)
            if 7 in tile_record_x:
                tile_record_x.remove(7)
            if 8 in tile_record_x:
                tile_record_x.remove(8)
            if 14 in tile_record_x:
                tile_record_x.remove(14)
            if 15 in tile_record_x:
                tile_record_x.remove(15)

        if (12 in user_data['latest_decision']):
            if 1 in tile_record_x:
                tile_record_x.remove(1)
            if 5 in tile_record_x:
                tile_record_x.remove(5)
        if (13 in user_data['latest_decision']):
            if 2 in tile_record_x:
                tile_record_x.remove(2)
            if 6 in tile_record_x:
                tile_record_x.remove(6)
        if (14 in user_data['latest_decision']):
            if 3 in tile_record_x:
                tile_record_x.remove(3)
            if 7 in tile_record_x:
                tile_record_x.remove(7)
        if (15 in user_data['latest_decision']):
            if 4 in tile_record_x:
                tile_record_x.remove(4)
            if 8 in tile_record_x:
                tile_record_x.remove(8)

        if (all(x in tile_record_x for x in [2, 3, 6, 7])) and (not any(x in user_data['latest_decision'] for x in [2, 3, 6, 7, 13, 14])):
                tile_record_x.remove(2)
                tile_record_x.remove(3)
                tile_record_x.remove(6)
                tile_record_x.remove(7)
                tile_record_x.append(10)

        elif (all(x in tile_record_x for x in [1, 2, 5, 6])) and (not any(x in user_data['latest_decision'] for x in [1, 2, 5, 6, 12, 13])):
                tile_record_x.remove(1)
                tile_record_x.remove(2)
                tile_record_x.remove(5)
                tile_record_x.remove(6)
                tile_record_x.append(9)
            

        elif (all(x in tile_record_x for x in [3, 4, 7, 8])) and (not any(x in user_data['latest_decision'] for x in [3, 4, 6, 7, 14, 15])):
                tile_record_x.remove(3)
                tile_record_x.remove(4)
                tile_record_x.remove(7)
                tile_record_x.remove(8)
                tile_record_x.append(11)
            
                

        if (all(x in tile_record_x for x in [1, 5])):
                tile_record_x.remove(1)
                tile_record_x.remove(5)
                tile_record_x.append(12)

        if (all(x in tile_record_x for x in [2, 6])):
                
                tile_record_x.remove(2)
                tile_record_x.remove(6)
                tile_record_x.append(13)
                
        if (all(x in tile_record_x for x in [3, 7])):
                tile_record_x.remove(3)
                tile_record_x.remove(7)
                tile_record_x.append(14)
                
        if (all(x in tile_record_x for x in [4, 8])):
                
                tile_record_x.remove(4)
                tile_record_x.remove(8)
                tile_record_x.append(15)
                
        if ((1 in user_data['latest_decision']) or (5 in user_data['latest_decision'])) and 12 in tile_record_x:
            tile_record_x.append(1)
            tile_record_x.append(5)
            tile_record_x.remove(12)

        if ((2 in user_data['latest_decision']) or (6 in user_data['latest_decision'])) and 13 in tile_record_x:
            tile_record_x.append(2)
            tile_record_x.append(6)
            tile_record_x.remove(13)

        if ((3 in user_data['latest_decision']) or (7 in user_data['latest_decision'])) and 14 in tile_record_x:
            tile_record_x.append(3)
            tile_record_x.append(7)
            tile_record_x.remove(14)

        if ((4 in user_data['latest_decision']) or (8 in user_data['latest_decision'])) and 15 in tile_record_x:
            tile_record_x.append(4)
            tile_record_x.append(8)
            tile_record_x.remove(15)

    
    tile_record_x.append(-1)

    return tile_record_x


def tile_decision3(predicted_record, video_size, range_fov, chunk_idx, user_data):
    """
    Deciding which tiles should be transmitted for each chunk, within the prediction window
    (As an example, users can implement their customized function.)

    Parameters
    ----------
    predicted_record: dict
        the predicted motion, with format {yaw: , pitch: , scale:}, where
        the parameter 'scale' is used for transcoding approach
    video_size: dict
        the recorded whole video size after video preprocessing
    range_fov: list
        degree range of fov, with format [height, width]
    chunk_idx: int
        index of current chunk
    user_data: dict
        user related data structure, necessary information for tile decision

    Returns
    -------
    tile_record: list
        the decided tile list of current update, each item is the chunk index
    """
    # The current tile decision method is to sample the fov range corresponding to the predicted motion of each chunk,
    # and the union of the tile sets mapped by these sampling points is the tile set to be transmitted.

    #if (user_data['video_analysis'][chunk_idx]['data'] == None):
    #    user_data['video_analysis'][chunk_idx]['background_tile_id'] = f"chunk_{str(chunk_idx).zfill(4)}_background"
    #    tile_idx = config_params['total_tile_num']
    #    tile_id = f"chunk_{str(chunk_idx).zfill(4)}_tile_{str(tile_idx).zfill(3)}"
    #    user_data['video_analysis'][chunk_idx]['tile_id'] = tile_id
    #    predicted_point = predicted_record[0]
    #    _3d_polar_coord = fov_to_3d_polar_coord([float(predicted_motion['yaw']), float(predicted_motion['pitch']), 0], range_fov, [600, 600])
    #    pixel_coord = pixel_coord_to_tile_corrected(_3d_polar_coord, config_params['projection_mode'], [converted_height, converted_width])
        
    try:
        curr_path = os.path.dirname(os.path.abspath(__file__)) 
        json_path=os.path.join(curr_path,"tile_prediction.json")
        result_chunk_name = f"chunk_{str(chunk_idx)}"
        tile_prediction =  read_tile_prediction_json(json_path)
    except:
        tile_prediction = [1,2,3,4,5,6,7,8]
    
    config_params = user_data['config_params']
    tile_record = []
    sampling_size = [50, 50]
    converted_width = user_data['config_params']['converted_width']
    converted_height = user_data['config_params']['converted_height']
    for predicted_motion in predicted_record:
        _3d_polar_coord = fov_to_3d_polar_coord([float(predicted_motion['yaw']), float(predicted_motion['pitch']), 0], range_fov, sampling_size)
        pixel_coord = _3d_polar_coord_to_pixel_coord(_3d_polar_coord, config_params['projection_mode'], [converted_height, converted_width])
        coord_tile_list = pixel_coord_to_tile_corrected_decision_making(pixel_coord, config_params['total_tile_num'], video_size, chunk_idx)
        unique_tile_list = [int(item) for item in np.unique(coord_tile_list)]
        tile_record.extend(unique_tile_list)

    if 0 in tile_record:
        tile_record.remove(0)
    if (config_params['total_tile_num']-1) in tile_record:
        tile_record.remove(config_params['total_tile_num']-1)

    final_tile_set = [i for i in tile_record if i in tile_prediction[result_chunk_name]['Predicted_tiles']]

    tile_record_x = [x for x in final_tile_set]
    
    
    tile_record_x.append(-1)

    return tile_record_x
