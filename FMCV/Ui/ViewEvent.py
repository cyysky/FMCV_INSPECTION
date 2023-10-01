import traceback
from threading import Lock

_lock = Lock()

from FMCV.Cv import Cv

def init(s):
    global start
    global off_live
    global live
    
    start = s 
    
    live = s.config.MODE.show_live
    off_live = not s.config.MODE.show_live
    
live = True
off_live = False
source = False

def update_source(in_source):
    global source
    with _lock:        
        source = in_source
        
        if not source:  
            return
        
        ret, roi = start.Profile.get_selected_roi()
        if not ret:
            source = False
            return 
        
        if roi.get('source') is None:
            source = False
            return
        
        ret , rois = start.Profile.get_roi_by_name(roi.get('source'))
        if not ret:
            source = False
            return
        
        start.MainUi.view.scale = 1 
        start.MainUi.view.set_rotate(roi.get('rotate'))
        if rois[0].get("img") is None:            
            start.MainUi.view.set_image(Cv.get_white_image_with_text(f'Please Set ROI {roi["name"]} Image'))
            return 
        start.MainUi.view.set_image(rois[0].get("img"))
        return

def update_view():
    global live
    global off_live

    if live:
        try:
            with _lock:
                if not source:
                    frames = start.Cam.get_live()    
                    start.MainUi.update_source(frames)
                    start.MainUi.frm_cams.update(frames)
        except:
            traceback.print_exc()
        start.MainUi.view.after(33, update_view)
        
    if off_live:
        start.MainUi.view.scale = 1
        start.MainUi.view.set_image(Cv.get_white_image_with_text('Live view off for performance'))
        

        
 