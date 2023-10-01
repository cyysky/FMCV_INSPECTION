import cv2
import traceback
import copy 
import re


import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

try:
    import easyocr
    easyocr_reader = easyocr.Reader(['en'],gpu=False)
except:
    traceback.print_exc()
    
if 'easyocr_reader' in globals():
    print("easyocr loaded successfully")

def process_ocr(start, result, frm, src_n, step_n, roi_n):

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
        
        res,top_left,bottom_right,top_val,cropped = start.Cv.match_template(roi['img'], frm[y1:y2,x1:x2] ,BlurSize=roi['blur'])
        result.update({'search_score':top_val})
        result.update({'offset_x':top_left[0] - xm})
        result.update({'offset_y':top_left[1] - ym})
    else:    
        cropped = frm[y1:y2,x1:x2]
    
    cropped_gray = cropped
    
    if len(cropped.shape) ==3:
        cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    if 'easyocr_reader' in globals():
        ocr_results = easyocr_reader.readtext(cropped)
        ocr_string = ""
        for ocr_text in ocr_results:
            print(ocr_text)
            ocr_string += str(ocr_text[-2])
    else:
        ocr_string = pytesseract.image_to_string(cropped_gray)

    result.update({'OCR':ocr_string})
    
    re_results = re.findall(roi.get("re"), ocr_string)
    
    if len(re_results) > 0:
        result.update({"PASS":True})
    else:
        result.update({"PASS":False})
        
    result.update({'result_image':copy.deepcopy(cropped)})
    result.update({'score':len(re_results)})
    
    return result