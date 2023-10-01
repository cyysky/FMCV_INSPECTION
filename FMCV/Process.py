# #==========================================================================
# import hashlib
# from FMCV.peace import Peace,license
# lic = license()['Vision']
# if (hashlib.sha3_256((lic+lic).encode('utf-8')).hexdigest()!=Peace(lic)):
    # # https://stackoverflow.com/questions/9555133/e-printstacktrace-equivalent-in-python
    # #traceback.print_exc()
    # # https://stackoverflow.com/questions/73663/terminating-a-python-script
    # import os
    # import sys
    # os._exit(0)
    # sys.exit(0)
    # raise SystemExit  
    # 1/0      
    # pass
# #==========================================================================

import cv2
import copy
import traceback

from FMCV.Processor import Barcode
from FMCV.Processor import Fiducial
from FMCV.Processor import Ano
from FMCV.Processor import Ocr
from FMCV.Processor import CCD3D
from FMCV.Processor import Locate
from FMCV.Processor import Distance
from FMCV.Cv import Cv

import numpy as np

from FMCV import Logging

def init(s):
    global start
    start = s 

def execute(frm, roi, src_n, step_n, roi_n):
    h,w = frm.shape[:2]
    #roi = start.Main.results[src_n][step_n][roi_n]
    #roi = copy.deepcopy(start.Profile.loaded_profile[src_n][step_n]['roi'][roi_n])
    result = roi
    
    if roi['type'] in ("AI","CNN"):
        m = roi['margin']
        x1 = roi['x1'] - m
        y1 = roi['y1'] - m
        x2 = roi['x2'] + m
        y2 = roi['y2'] + m
        Logging.info(f"x1:y1={x1}:{y1} x2:y2={x2}:{y2} w:h={x2-x1}:{y2-y1}")
        if m > 0:
            xm = m
            ym = m
    
            if x1 < 0 : xm = m + x1
            if y1 < 0 : ym = m + y1
                
            if x1 < 0 : x1 = 0 
            if y1 < 0 : y1 = 0 
            if x2 > w : x2 = w 
            if y2 > h : y2 = h

            res,top_left,bottom_right,top_val,cropped = start.Cv.match_template(roi['img'], frm[y1:y2,x1:x2] ,BlurSize=roi['blur'])
            result.update({'search_score':top_val})
            result.update({'offset_x':top_left[0] - xm})
            result.update({'offset_y':top_left[1] - ym})
        else:
            cropped = frm[y1:y2,x1:x2]
            
        classify, score, blended_heatmap = start.CNN.predict("go",cropped)
        result_name = start.CNN.get_class_name(classify)
                
        if roi["class"] == result_name :
            if score < start.Config.ai_minimum:
                result.update({"PASS":False})
            else:
                result.update({"PASS":True})
                
            if score < roi["minimal"]:
                result.update({"PASS":False})
            else:
                result.update({"PASS":True})
                
        else:
            result.update({"PASS":False})
        
        if blended_heatmap is not None:
            result.update({'result_blended_heatmap':copy.deepcopy(blended_heatmap)})
            result.update({'result_image':copy.deepcopy(cropped)})
        else:
            result.update({'result_image':copy.deepcopy(cropped)})
        if start.Config.ai_detail:
            result.update({"result_class":result_name})
        else:
            if roi["class"] == result_name :
                result.update({"result_class":"PASS"})
            else:
                result.update({"result_class":"FAIL"})
            
        result.update({"result_score":score})
        Logging.info(f'{roi["name"]} {result_name}:{score}')
    
    if roi['type'] in ("AI2","ANO"):
        result = Ano.process_ano(start, result, frm, src_n, step_n, roi_n)
    
    if roi['type'] == "QR":
        result = Barcode.process_barcode(start, result, frm, src_n, step_n, roi_n)
    
    if roi['type'] in ("2D", "FIDUCIAL"):
        result = Fiducial.process_fiducial(start, result, frm, src_n, step_n, roi_n)
        
    if roi['type'] == "OCR":
        result = Ocr.process_ocr(start, result, frm, src_n, step_n, roi_n)
        
    if roi['type'] == "3D":
        result = CCD3D.process_3D(start, result, frm, src_n, step_n, roi_n)
        
    if roi['type'] == "MOVE":
        result.update({"PASS":True})
        
    if roi['type'] == "LOC":
        Locate.locate(start, result, frm, src_n, step_n, roi_n)
        
    if roi['type'] == "DIST":
        Distance.distance(start, result, frm, src_n, step_n, roi_n)
                        
    return result