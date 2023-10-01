from FMCV.Cv import Match,Cv
import copy 

def locate(start, result, frm, src_n, step_n, roi_n):
    roi = result
    
    result.update({"PASS":False})
    
    a = Match.match_rotate(roi.get('img'),frm.copy(),disable_rectangle_check = True, blur = roi.get("blur"))
    
    height,width = roi.get('img').shape[:2]
    
    if not a:
        result.update({'result_image':Cv.get_white_image_with_text('',image_height = height , image_width = width)})
        return
    
    frame, cropped_r, [xm_center, ym_center], angle , dst, xy_dst = a
    
    center = (xm_center, ym_center)
    
    rotated_image = Match.extract_rotated_rectangle(frm.copy(), center, width, height, angle)
    
    result.update({"PASS":True})
    result.update({'result_image':copy.deepcopy(cropped_r)})
    