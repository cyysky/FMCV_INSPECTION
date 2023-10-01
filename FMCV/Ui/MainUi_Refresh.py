from FMCV.Ui import MainUi as M
from tkinter import *
self = M.self
import base64
import copy
import cv2
import traceback
import numpy as np
import math 
from FMCV.Cv import Cv
import os
from FMCV import Logging

def update_cursor(event):
 
    #if not self.Users.login_admin():
    #    return
        
    ret, roi = self.Profile.get_selected_roi()
    if not ret:
        return
        
    scale = M.view.get_scale()
    
    x1 = roi['x1'] * scale
    y1 = roi['y1'] * scale
    x2 = roi['x2'] * scale
    y2 = roi['y2'] * scale

    img_x, img_y = (M.view.viewer.canvasx(event.x), M.view.viewer.canvasy(event.y))
    
    zp = 5 # resize activate zone pixel
    M.view.viewer.config(cursor="top_left_arrow")
    
    if x2+ zp>= img_x > x2- zp and y2+ zp > img_y > y2- zp and M.cf.move_all_var.get():
        M.view.viewer.config(cursor="lr_angle")

    elif x2- zp >= img_x >= x1 and y2- zp >= img_y >= y1 and M.cf.move_all_var.get():
        M.view.viewer.config(cursor="fleur")
        
    elif x2- zp >= img_x >= x1 and y2- zp >= img_y >= y1:
        M.view.viewer.config(cursor="fleur")
        
    elif x2+ zp>= img_x > x2 - zp and y2+ zp > img_y > y2 - zp:
        M.view.viewer.config(cursor="lr_angle")


def display_other_roi(*args):
    M.view.viewer.delete("other")
    ret, roi = self.Profile.get_selected_roi()
    if ret:
        scale = M.view.get_scale()
        for roi_n, roi_aux in enumerate(self.Profile.loaded_profile[M.cam_pos][M.cmb_pos]["roi"]):
            if M.roi_index == roi_n:
                continue
            M.view.viewer.create_rectangle(roi_aux['x1']* scale, roi_aux['y1']* scale, roi_aux['x2']* scale, roi_aux['y2'] * scale , outline="gray", tags="other")
        
def display_roi_margin(*args):
    M.view.viewer.delete("other")
    ret, roi = self.Profile.get_selected_roi()
    if ret and  roi.get("margin") > 0:
        scale = M.view.get_scale()
        m = roi.get("margin")
        M.view.viewer.create_rectangle((roi['x1']-m) * scale, (roi['y1']-m)* scale, (roi['x2']+m)* scale, (roi['y2']+m) * scale , outline="gray", tags="other")
    

def refresh_edit_roi_rectangle():
    if M.roi_index > -1:
        #print(f'step_cmb_pos {M.cmb_pos} roi_cmb_pos {M.roi_index}')
        scale = M.view.get_scale()
        roi = self.Profile.loaded_profile[M.cam_pos][M.cmb_pos]["roi"][M.roi_index]
        move_rectangle(roi['x1']*scale, roi['y1']*scale, roi['x2']*scale, roi['y2']*scale)
    else:
        remove_rectangle()
        
def remove_rectangle():
    move_rectangle(-1,-1,-1,-1)
    M.view.viewer.delete("other")
    
def move_rectangle(x1, y1, x2, y2):
    if M.id_rect == -1:
        M.id_rect = M.view.viewer.create_rectangle(x1, y1, x2, y2,outline='orange')
        print(f'id_rect = {M.id_rect}')
    else:
        M.view.viewer.coords(M.id_rect, x1, y1, x2, y2)

def refresh_listbox(pos = -1):
    M.event_unbind()
    remove_rectangle()
    
    # Refresh Step Combobox
    refresh_step_cmb()
    
    # Refresh Source Combobox
    refresh_source_cmb()
    
    if pos == -1:
        pos = M.cmb_pos
    M.Lb1.delete(0,END)
    try:        
        for roi_n, roi in enumerate(self.Profile.loaded_profile[M.cam_pos][pos]["roi"]):
            M.Lb1.insert(roi_n, roi['name'])
            
        if (len(self.Profile.loaded_profile[M.cam_pos][pos]["roi"]) - 1) <  M.roi_index:
            M.roi_index = len(self.Profile.loaded_profile[M.cam_pos][pos]["roi"]) -1
        M.Lb1.select_set(M.roi_index)    
        M.Lb1.activate(M.roi_index)
        
        M.Lb1.see(M.roi_index) #https://stackoverflow.com/questions/10155153/how-to-scroll-to-a-selected-item-in-a-scrolledlistbox-in-python
    except:        
        M.roi_index = -1
        print("Empty source and position")
    M.event_bind()

def refresh_result_view(pos = -1):
    if pos == -1:
        pos = M.cmb_pos
    M.r_view.viewer.delete("all")

    try:
        #M.r_view.current_results = self.Main.results[M.cam_pos][pos]
        if M.cam_pos < len(self.Main.result_frame):
            if pos < len(self.Main.result_frame[M.cam_pos]):
                if isinstance(self.Main.result_frame[M.cam_pos][pos][""],np.ndarray):
                    M.r_view.set_image(self.Main.result_frame[M.cam_pos][pos][""])
                    scale = M.r_view.get_scale()
                    
                    h, w = self.Main.result_frame[M.cam_pos][pos][""].shape[:2]
                    
                    for roi_n, roi in enumerate(self.Main.results[M.cam_pos][pos]):
                    
                        roi_pass = roi.get('PASS')
                        if roi_pass is True:
                            color = 'green'
                        else:
                            color = 'red'
                        
                        off_x = 0
                        off_y = 0
                        
                        if roi.get("offset_x") is not None:
                            off_x = roi["offset_x"]
                            
                        if roi.get("offset_y") is not None:
                            off_y = roi["offset_y"]
                        
                        
                        x1 = roi['x1'] + off_x
                        y1 = roi['y1'] + off_y
                        x2 = roi['x2'] + off_x
                        y2 = roi['y2'] + off_y
                        
                        degree_name = self.Main.results[M.cam_pos][pos][roi_n]['rotate']
                        x1, y1, x2, y2 = Cv.get_rotate_rectangle(degree_name, (w, h), (x1, y1, x2, y2))
                        
                        M.r_view.viewer.create_rectangle(x1 * scale, y1 * scale, x2 * scale, y2 * scale , outline=color, tags=(str(roi_n)))
    except:
        traceback.print_exc()
        print("refresh_result_view: didn't have results ")        

def rotatebox( rect, center, degrees ):
    rads = math.radians(degrees)

    newpts = []
    for pts in rect:
        diag_x = center[0] - pts[0]
        diag_y = center[1] - pts[1]

        # Rotate the diagonal from center to top left

        newdx = diag_x * math.cos(rads) - diag_y * math.sin(rads)
        newdy = diag_x * math.sin(rads) + diag_y * math.cos(rads)
        newpts.append( (center[0] + newdx, center[1] + newdy) )

    return newpts

def rotaterectcw( rect, center ):
    x0 = rect[0] - center[0]
    y0 = rect[1] - center[1]
    x1 = rect[2] - center[0]
    y1 = rect[3] - center[1]
    return center[0]+y0, center[1]-x0, center[0]+y1, center[1]-x1
        
def rotate90Deg(bndbox, img_width): # just passing width of image is enough for 90 degree rotation.
   x_min,y_min,x_max,y_max = bndbox
   new_xmin = y_min
   new_ymin = img_width-x_max
   new_xmax = y_max
   new_ymax = img_width-x_min
   return [new_xmin, new_ymin,new_xmax,new_ymax]

        
def refresh_step_cmb():
    try:
        step_list = []
        for step_n, step in enumerate(self.Profile.loaded_profile[M.cam_pos]):
            step_list.append(step_n+1)
        M.cmb['values'] = tuple(step_list)
        M.cmb.current(M.cmb_pos)
    except:
        print("Empty step of source")
        
def refresh_source_cmb():
    source_list = []
    for src_n, src in enumerate(self.Profile.loaded_profile):
        source_list.append(src_n+1)
    if len(source_list) > 0:
        M.cmb_cam['values'] = tuple(source_list)
    else:
        M.cmb_cam['values'] = tuple([1])
    M.cmb_cam.current(M.cam_pos)

def refresh_profile_cmb():
    M.cmb_profile['values'] = tuple(next(os.walk("Profile"))[1])
    M.cmb_profile.set(self.Profile.name)
    self.Config.config.PROFILE.name = self.Profile.name
    self.Config.profile = self.Profile.name
    
    title = "FMCV AI Deep Learning Vision Inspection"
    
    M.top.title(f"{title} version {self.Config.version}, Selected Profile : {self.Config.profile}")

def display_roi_image():
    M.roi_setting_frame.image_view.clear()
    if M.roi_index > -1:
        roi = self.Profile.loaded_profile[M.cam_pos][M.cmb_pos]["roi"][M.roi_index]
        if roi.get('img') is not None:
            #print("display image")
            M.roi_setting_frame.image_view.scale = 1
            M.roi_setting_frame.image_view.set_image(roi['img'])
        

def refresh_main_ui():
    # Refresh border color of live stream
    if self.ViewEvent.live:
        self.MainUi.view.set_border_color("default")
    else:
        self.MainUi.view.set_border_color("brown")

    # Refresh Step Display at Control Panel
    self.MainUi.steps_lbl.config(text = "Steps = {}".format(self.Main.detected_step + 1))
    
    # Refresh Step combobox
    self.MainUi.cmb_pos = self.Main.detected_step
    
    # Refresh Roi Listbox
    refresh_listbox() # Have nested call   refresh_step_cmb() & refresh_source_cmb()
    
    if self.MainUi.cf.move_all_var.get():
        display_other_roi()
    
    # Refresh Results
    refresh_result_view(self.Main.detected_step)

    # Refresh Camera Results Box
    self.MainUi.frm_cams.update_results()

    # Refresh Results Status
    if self.Config.show_running == False:
        self.MainUi.result_frame.set_result(self.Main.is_step_pass(self.MainUi.cmb_pos))
    
    # Refresh ROI setting Panel
    self.MainUi.refresh_roi_index() #Nested call refresh_edit_roi_rectangle() and display_roi_image()#
    
    # Refresh related ROI result
    if M.roi_index > -1:
        try:
            result_roi = self.Main.results[M.cam_pos][M.cmb_pos][M.roi_index]
            self.MainUi.result_frame.update_results(result_roi)#self.MainUi.r_view.current_results[M.roi_index])
            M.result_frame.result_roi = result_roi
        except:
            Logging.info("Didn't have Results")
    else:
        M.result_frame.result_roi = {}
        self.MainUi.result_frame.update_results({})
    
    # Refresh Operation Results 
    self.OperationResults.refresh()
    
    #Refresh profile combobox
    refresh_profile_cmb()

    #Refresh done successfully


M.view.viewer.bind("<Motion>",update_cursor)


#refresh_cmb_cam()
refresh_source_cmb()
refresh_step_cmb()
refresh_listbox()
refresh_profile_cmb()

