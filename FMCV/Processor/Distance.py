from FMCV import Logging
from FMCV.Cv import Match,Cv

import copy 
import numpy as np

def distance(start, result_roi, frm, src_n, step_n, roi_n):
    
    roi = result_roi
    
    result_roi.update({"PASS":False})
    
    height,width = frm.shape[:2]
    
    m = roi['margin']
    
    x1 = roi['x1'] - m
    y1 = roi['y1'] - m
    x2 = roi['x2'] + m
    y2 = roi['y2'] + m
    
    mm_pixel = roi.get('mm_pixel')
    
    #Logging.info(f"{roi['name']} x1 {x1} y1 {y1} x2 {x2} y2 {y2}")
    
    xm = m
    ym = m
    
    if x1 < 0 : xm = m + x1
    if y1 < 0 : ym = m + y1
    
    if x1 < 0 : x1 = 0 
    if y1 < 0 : y1 = 0 
    if x2 > width : x2 = width 
    if y2 > height : y2 = height
    
    if np.all([roi['mask'] == 255]):
        mask = None
        Logging.info("White template")
    else:
        mask = roi['mask']
        Logging.info("Masked template") 
    
    refer_offset_x = 0
    refer_offset_y = 0
    refer_x = roi['x1']
    refer_y = roi['y1']
    
    if roi.get("refer") != "":        
        ret, result_rois = start.Main.get_results_with_result_image_by_name(roi.get("refer"))
        if ret:
            refer_x = result_rois[0]['x1']
            refer_y = result_rois[0]['y1']
            refer_offset_x = result_rois[0]['offset_x']
            refer_offset_y = result_rois[0]['offset_y']
            Logging.info("refer x y offset x y",refer_x,refer_y,refer_offset_x,refer_offset_y)
    
    res, top_left, bottom_right, top_val, cropped = start.Cv.match_template(roi['img'] , frm[y1:y2,x1:x2] , 
                                                                            Mask=mask, BlurSize=roi['blur'], Mode=roi['method'])
    
    offset_x = top_left[0] - xm
    offset_y = top_left[1] - ym
    
    distance = start.Match.distance((refer_x + refer_offset_x, refer_y + refer_offset_y), (roi['x1'] + offset_x, roi['y1'] + offset_y))
    distance_x = abs((refer_x + refer_offset_x) - (roi['x1'] + offset_x))
    distance_y = abs((refer_y + refer_offset_y) - (roi['y1'] + offset_y))
    
    result_roi.update({'offset_x': offset_x})
    result_roi.update({'offset_y': offset_y})
    result_roi.update({'distance': distance})
    result_roi.update({'distance_x': distance_x})  
    result_roi.update({'distance_y': distance_y})
    
    result_roi.update({'offset_x_mm':offset_x * mm_pixel})
    result_roi.update({'offset_y_mm':offset_y * mm_pixel})
    result_roi.update({'distance_mm': distance * mm_pixel})
    result_roi.update({'distance_x_mm': distance_x * mm_pixel})
    result_roi.update({'distance_y_mm': distance_y * mm_pixel})

    distance_max = roi['distance_max']
    distance_min = roi['distance_min']
    distance_y_max = roi['distance_y_max']
    distance_y_min = roi['distance_y_min']
    distance_x_max = roi['distance_x_max']
    distance_x_min = roi['distance_x_min']

    # basic judgement
    if (roi.get('minimal') <= top_val) and \
       (distance_min <= (distance * mm_pixel) <= distance_max) and \
       (distance_y_min <= (distance_y * mm_pixel) <= distance_y_max) and \
       (distance_x_min <= (distance_x * mm_pixel) <= distance_x_max):
        Logging.info("All distances are within the specified range.")
        result_roi.update({"PASS":True})
    else:
        Logging.info("One or more distances are out of the specified range.")
        result_roi.update({"PASS":False})

    result_roi.update({'result_image':copy.deepcopy(cropped)})
    result_roi.update({'score':top_val})
