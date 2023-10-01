import os
import traceback
import cv2
from datetime import datetime, timedelta
import time
from FMCV import Logging
import xml.etree.ElementTree as ET
import sqlite3
from pathlib import Path


def init(start):
    global s
    global barcode, mes_path, results_path, images_path, log_datetime, results, result_frame, log_datetime_iso8601
    
    s = start
    
    barcode = s.Log.barcode
    mes_path = s.Log.mes_path
    results_path = s.Log.results_path
    images_path = s.Log.images_path
    log_datetime = s.Log.log_datetime
    results = s.Log.results
    result_frame = s.Log.result_frame
    log_datetime_iso8601 = s.Log.log_datetime_iso8601
    
def write_log():
    global images_path
    
    if str(results_path) != '.':
        os.makedirs(results_path / "CSV", exist_ok=True)
        results_file_path = results_path / "CSV" / "{}_{}.txt".format(barcode,log_datetime)
        cycle_time = time.time()-s.Main.cycle_start_time
        
        overall_result = True
        
        with open(results_file_path, "w") as file_log:    
            file_log.write("Barcode,Camera,Step,ROI_Name,Type,Result,Aux1,Aux2,Aux3,Aux4,Aux5,Aux6\n")
            for src_n, src in enumerate(results):     
                for step_n, step in enumerate(results[src_n]):
                    for roi_n, roi_result in enumerate(results[src_n][step_n]):
                        file_log.write("{}".format(barcode))
                        file_log.write(",")
                        file_log.write("{}".format(src_n+1))
                        file_log.write(",")
                        file_log.write("{}".format(step_n+1))
                        file_log.write(",")
                        file_log.write("{}".format(roi_result["name"]))
                        file_log.write(",")
                        file_log.write("{}".format(roi_result.get("type")))
                        file_log.write(",")
                        if roi_result.get("PASS"):
                            state = "PASS"
                        else:
                            state = "FAIL"
                            overall_result = False
                        file_log.write("{}".format(state))
                        file_log.write(",")
                        if roi_result.get("type") in("AI","CNN"):
                            file_log.write("{}".format(roi_result.get("result_class")))
                            file_log.write(",")
                            file_log.write("{}".format(roi_result.get("result_score")))
                        
                        if roi_result.get("type") in("AI2","ANO"):
                            file_log.write("{}".format(roi_result.get("result_score")))
                        
                        if roi_result.get("type") == "QR":
                            file_log.write("{}".format(roi_result.get("CODE")))
                            
                        if roi_result.get("type") in("2D","FIDUCIAL"):
                            file_log.write("{},".format(roi_result.get("score")))
                            file_log.write("{},".format(roi_result.get("offset_x")))
                            file_log.write("{},".format(roi_result.get("offset_y")))
                            file_log.write("{},".format(roi_result.get("angle")))
                            file_log.write("{},".format(roi_result.get("offset_x_mm")))
                            file_log.write("{}".format(roi_result.get("offset_y_mm")))
                            
                        if roi_result.get("type") == "OCR":
                            file_log.write("{},".format(roi_result.get("OCR")))
                            
                        file_log.write("\n")
                        
            file_log.write(",,,RESULT,{},CYCLE TIME,{},\n".format("PASS" if overall_result else "FAIL",cycle_time))
    
        
            os.makedirs(results_path / "XML", exist_ok=True)
            results_file_path = results_path / "XML" / "{}_{}.xml".format(barcode,log_datetime)
            # Create the root element of the XML document
            root = ET.Element('FMCV_H')
            barcode_element = ET.SubElement(root, 'BARCODE')
            barcode_element.set('name', str(barcode))
            for src_n, src in enumerate(results):   
                # Create a child element for each sources
                sources = ET.SubElement(root, 'SOURCE')
                sources.set('name', str(src_n+1))
                for step_n, step in enumerate(results[src_n]):
                    steps = ET.SubElement(sources, 'STEP')
                    steps.set('name', str(step_n+1))
                    for roi_n, roi_result in enumerate(results[src_n][step_n]):
                        rois = ET.SubElement(steps, 'ROI')
                        rois.set('name', str(roi_n+1))
                        ET.SubElement(rois, 'name').text = str(roi_result["name"])
                        ET.SubElement(rois, 'type').text = str(roi_result.get("type"))
                        ET.SubElement(rois, 'PASS').text = str(roi_result.get("PASS"))
                        
                        if roi_result.get("type") in("AI","CNN"):
                            ET.SubElement(rois, 'result_class').text = str(roi_result.get("result_class"))
                            ET.SubElement(rois, 'result_score').text = str(roi_result.get("result_score"))
                            if roi_result.get("edit_user") is not None:
                                ET.SubElement(rois, 'edit_user').text = str(roi_result.get("edit_user"))
                            
                        if roi_result.get("type") == "QR":
                            ET.SubElement(rois, 'CODE').text = str(roi_result.get("CODE"))
                            
                        if roi_result.get("type") in ("2D","FIDUCIAL"):
                            ET.SubElement(rois, 'score').text = str(roi_result.get("score"))
                            ET.SubElement(rois, 'offset_x').text = str(roi_result.get("offset_x"))
                            ET.SubElement(rois, 'offset_y').text = str(roi_result.get("offset_y"))
                            ET.SubElement(rois, 'offset_x_mm').text = str(roi_result.get("offset_y_mm"))
                            ET.SubElement(rois, 'offset_y_mm').text = str(roi_result.get("offset_y_mm"))
                            ET.SubElement(rois, 'angle').text = str(roi_result.get("angle"))
                            
                        if roi_result.get("type") == "OCR":
                            ET.SubElement(rois, 'OCR').text = str(roi_result.get("OCR"))
                            
                        if roi_result.get("type") in ("AI2","ANO"):
                            ET.SubElement(rois, 'result_score').text = str(roi_result.get("result_score"))
                            
            if root is not None:                
                # Write the XML document to a file
                root = ET.ElementTree(root)
                ET.indent(root, space="\t", level=0)
                root.write(results_file_path, encoding='utf-8', xml_declaration=True)
                
            
    if str(images_path) != '.':
        images_path_2 = images_path / str(barcode +" "+log_datetime)
        os.makedirs(images_path_2, exist_ok=True)
        for src_n, src in enumerate(results):     
            for step_n, step in enumerate(results[src_n]):
                for roi_n, roi_result in enumerate(results[src_n][step_n]):    
                    if roi_result.get("PASS"):
                        state = "PASS"
                    else:
                        state = "FAIL"
                    try:
                        if str(images_path) != '.':
                            cv2.imwrite(str(images_path_2 / "{}_{}_{}_{}.jpg".format(barcode,f"{src_n+1}'{step_n+1}'{roi_result.get('name')}",state,log_datetime)), roi_result.get('result_image'))
                    except:
                        Logging.error("Result image not write ",f"{src_n+1}'{step_n+1}'{roi_result.get('name')}")
                        #traceback.print_exc()