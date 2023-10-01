import cv2
import traceback
import copy 
import numpy as np

def process_ano(start, result, frm, src_n, step_n, roi_n):

    h,w = frm.shape[:2]
    roi = result
    
    m = roi['margin']
    
    x1 = roi['x1'] - m
    y1 = roi['y1'] - m
    x2 = roi['x2'] + m
    y2 = roi['y2'] + m
    
    print(f"x1:y1={x1}:{y1} x2:y2={x2}:{y2} w:h={x2-x1}:{y2-y1}")
    if m > 0:
        xm = m
        ym = m
        
        if x1 < 0 : xm = m + x1
        if y1 < 0 : ym = m + y1
        
        if x1 < 0 : x1 = 0 
        if y1 < 0 : y1 = 0 
        if x2 > w : x2 = w 
        if y2 > h : y2 = h
        
        res,top_left,bottom_right,top_val,cropped = start.Cv.match_template(roi['img'], frm[y1:y2,x1:x2], BlurSize=roi['blur'])
        result.update({'search_score':top_val})
        result.update({'offset_x':top_left[0] - xm})
        result.update({'offset_y':top_left[1] - ym})
    else:    
        cropped = frm[y1:y2,x1:x2]
    
    images = []
    image = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)            
    images.append(image)
    
    score,output_image = start.ANO.inference(images,roi['minimal'])
    
    result.update({'ano_result_image':copy.deepcopy(output_image)})
    
    if roi.get('minimal') >= score[0].detach().numpy():
        result.update({"PASS":True})
    else:
        result.update({"PASS":False})
        
    result.update({'result_image':copy.deepcopy(cropped)})
    result.update({'score':score[0].detach().numpy()})
    
    return result