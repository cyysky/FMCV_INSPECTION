import cv2
import traceback
import copy 
import numpy as np
import sys
from FMCV import Logging
def process_fiducial(start, result, frm, src_n, step_n, roi_n):

    h,w = frm.shape[:2]
    roi = result
    
    if roi.get('K') and roi.get('D'):
        mtx = np.array(roi.get('K'))
        dist = np.array(roi.get('D'))
        newcameramtx, img_roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        dst = cv2.undistort(frm, mtx, dist, None, newcameramtx)
        ax, ay, aw, ah = img_roi
        dst = dst[ay:ay+ah, ax:ax+aw]
        frm = cv2.resize(dst, (w,h), interpolation = cv2.INTER_AREA)
        result.update({'2D_result_image':frm})
        
    m = roi['margin']
    
    x1 = roi['x1'] - m
    y1 = roi['y1'] - m
    x2 = roi['x2'] + m
    y2 = roi['y2'] + m
    
    print(f"x1 {x1} y1 {y1} x2 {x2} y2 {y2}")
    
    xm = m
    ym = m
    
    if x1 < 0 : xm = m + x1
    if y1 < 0 : ym = m + y1
    
    if x1 < 0 : x1 = 0 
    if y1 < 0 : y1 = 0 
    if x2 > w : x2 = w 
    if y2 > h : y2 = h
    

    if np.all([roi['mask'] == 255]):
        mask = None
        print("White template")
    else:
        mask = roi['mask']
        print("Masked template") 

    angle = -999
    FLOAT_MIN = float('-inf')
    INT_MIN = -sys.maxsize - 1
    top_val = FLOAT_MIN
    result.update({'offset_x':INT_MIN})
    result.update({'offset_y':INT_MIN})
    result.update({'offset_x_mm':FLOAT_MIN})
    result.update({'offset_y_mm':FLOAT_MIN})
    cropped = roi['mask']
    
    if roi.get("angle_enable"):
        a = start.Match.match_rotate(roi.get('img'),frm[y1:y2,x1:x2].copy())
        if a:
            frame, cropped_r, [xm_center, ym_center], angle , dst, xy_dst = a
            res,top_left,bottom_right,top_val,cropped = start.Cv.match_template(roi['img'] , cropped_r , Mask=mask, BlurSize=roi['blur'], Mode=roi['method'])
            
            result.update({'2D_result_image':frame})
            
            cx = int(frm.shape[1]/2)
            cy = int(frm.shape[0]/2)
            
            cx_roi = int( x1 + xm + int((roi['x2']-roi['x1'])/2))
            cy_roi = int( y1 + ym + int((roi['y2']-roi['y1'])/2))
            
            x_center = x1 + xm_center
            y_center = y1 + ym_center
            
            theta = -angle * np.pi / 180

            x_rot = cx + (x_center - cx)*np.cos(theta) - (y_center - cy)*np.sin(theta)
            y_rot = cy + (x_center - cx)*np.sin(theta) + (y_center - cy)*np.cos(theta)
            
            x_offset = x_rot - cx_roi
            y_offset = y_rot - cy_roi
            
            result.update({'offset_x':int(x_offset)})
            result.update({'offset_y':int(y_offset)})
            
    if result.get('offset_x') == INT_MIN and result.get("offset_y") == INT_MIN:
        res,top_left,bottom_right,top_val,cropped = start.Cv.match_template(roi['img'] , frm[y1:y2,x1:x2] , Mask=mask, BlurSize=roi['blur'], Mode=roi['method'])
        result.update({'offset_x':top_left[0] - xm})
        result.update({'offset_y':top_left[1] - ym})

    # basic judgement
    if roi.get('minimal') <= top_val:
        result.update({"PASS":True})
    else:
        result.update({"PASS":False})
    result.update({'result_image':copy.deepcopy(cropped)})
    result.update({'score':top_val})
    result.update({'angle':angle})
    # If 2d calibrated
    if roi.get('2d_calibrate') is not None:
        # If sift matched
        if angle>-999:
            off_x_mm, off_y_mm = start.Calibrate2d.get_offset_mm(x_offset, y_offset, roi['2d_calibrate'])
            result.update({'offset_x_mm':off_x_mm})
            result.update({'offset_y_mm':off_y_mm})
            
        # If templated matched
        elif result.get('offset_x') > FLOAT_MIN and result.get("offset_y") > FLOAT_MIN:
            off_x_mm, off_y_mm = start.Calibrate2d.get_offset_mm(top_left[0] - xm, top_left[1] - ym, roi['2d_calibrate'])
            result.update({'offset_x_mm':off_x_mm})
            result.update({'offset_y_mm':off_y_mm})
            
    # Non-2d_calibrated
    elif result.get("offset_x") > FLOAT_MIN and result.get("offset_y") > FLOAT_MIN:    
        result.update({'offset_x_mm':result.get('offset_x') * roi.get('mm_pixel')})
        result.update({'offset_y_mm':result.get('offset_y') * roi.get('mm_pixel')})
    #print(result.get("offset_x") , result.get("offset_x") > FLOAT_MIN)
    #Logging.info('margin={} x1={} y1={} topleft{} match {} pass {} pass {}'.format(m, x1, y1, top_left, top_val, roi.get('minimal') <= top_val, roi.get('minimal')))
    
    offset = {}
    offset.update({
                    "x":int(result.get("offset_x")) if result.get("offset_x")>FLOAT_MIN else 0, 
                    "y":int(result.get("offset_y")) if result.get("offset_y")>FLOAT_MIN else 0,
                    "x_mm":result.get("offset_x_mm") if result.get("offset_x_mm")>FLOAT_MIN else 0, 
                    "y_mm":result.get("offset_y_mm") if result.get("offset_y_mm")>FLOAT_MIN else 0, 
                    "angle":result.get("angle") if result.get("angle")>-999 else 0
                    })
    
    start.Com.update_offset(offset)
    
    if result['PASS'] == False:
        start.Com.failed()
    
    return result