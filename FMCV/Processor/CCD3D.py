import cv2
import traceback
import copy 
import numpy as np

from FMCV.Automata import Calibrate3d

def process_3D(start, result, frm, src_n, step_n, roi_n):

    roi = result
    
    cal_position ,img_chessboard, pixel_to_mm = Calibrate3d.calculate(roi.get('R'),roi.get('T'),roi.get('K'),roi.get('D'),  
                                start.Com.get_tcp(), offset_r_xyz = roi.get('offset_r_xyz'),
                                box_rows_cols = roi.get('box_rows_cols'), box_size = roi.get('box_size'), raw_frame = frm)
    
    result.update({'result_image':img_chessboard})
    if cal_position is None:
        result.update({"PASS":False})
    else:
        result.update({"PASS":True})  
        result.update({'3d_x':cal_position[0]})
        result.update({'3d_y':cal_position[1]})
        result.update({'3d_z':cal_position[2]})
        result.update({'3d_rx':cal_position[3]})
        result.update({'3d_ry':cal_position[4]})
        result.update({'3d_rz':cal_position[5]})
        
        
        start.Com.send_tcp_modbus(cal_position)
        
    if result['PASS'] == False:
        start.Com.failed()
    
    return result