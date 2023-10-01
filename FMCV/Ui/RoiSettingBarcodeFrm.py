from tkinter import *
from tkinter import messagebox
from tkinter import ttk
from tkinter import filedialog

import copy

import numpy
import base64
import cv2

from FMCV.Cv import Cv, Filter

class ROISettingBarcodeFrame(ttk.Frame):

    def __init__(self, start, parent, *args, **kwargs):
        #https://stackoverflow.com/questions/17466561/best-way-to-structure-a-tkinter-application/17470842
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.start = start

        Label(self, text = 'Width(pixel)').pack()
        self.qr_size = qr_size = Entry(self)
        qr_size.pack()
        qr_size.bind('<Return>',self.save)
        
        Label(self, text = 'Filter').pack()
        self.qr_filter = ttk.Combobox(self,state="readonly")
        self.qr_filter['values'] = ("","PCB_1","EQUALIZE","BINARY")
        self.qr_filter.pack()
        
        # create a BooleanVar to hold the state of the checkbox
        self.matrix_var = matrix_var = BooleanVar()

        # create a Checkbutton
        matrix_check_button = ttk.Checkbutton(self, text="Enable 2d Matrix", variable=matrix_var)
        matrix_check_button.pack()
        
        save_roi_image = Button(self,text ="View/Save Filter ROI")
        save_roi_image.pack(side=TOP)
        save_roi_image.configure(command=self.update_roi_image) 
        
        btn_save = Button(self,text ="Save")
        btn_save.pack(side=TOP)
        btn_save.configure(command=self.save)  
        
        if start.Config.mode_name != "ENGINEER":
            btn_save["state"] = DISABLED
        
    def update_roi_image(self):
        if not self.start.Users.login_admin():
            return
        
        ret, roi =  self.start.Profile.get_selected_roi()
        if not ret:
            return

        frame = self.start.Cam.get_current_image()
        x1 = roi['x1'] 
        y1 = roi['y1'] 
        x2 = roi['x2'] 
        y2 = roi['y2']
                
        #roi.update({"image":base64.b64encode(cv2.imencode('.png',copy.deepcopy(frame[y1:y2,x1:x2]))[1]).decode()})
                
        roi_img = copy.deepcopy(frame[y1:y2,x1:x2])
        
        selected_value = self.qr_filter.get()

        if selected_value == "PCB_1":
            roi_img = Filter.pcb_1(roi_img)
        elif selected_value == "BINARY":
            roi_img = Filter.ostu_binary(roi_img)
        elif selected_value == "EQUALIZE":
            roi_img = Filter.CLAHE(roi_img)
        if roi['qr_size'] > 10:        
            roi_img = Cv.resize_maintain_ratio_by_width(roi_img, roi['qr_size'])
        
        roi.update({'img':roi_img})
        self.start.Main.flag_reload = True
        self.start.Profile.flag_save = True   
        self.start.MainUi_Refresh.refresh_main_ui()  
    
    def save(self):
        if not self.start.Users.login_admin():
            return
        
        ret, roi =  self.start.Profile.get_selected_roi()
        if not ret:
            messagebox.showwarning("Warning","Please select roi on left")
            return
            
        try:            
            print(f"qr size {int(self.qr_size.get())}")
            roi.update({"qr_size":int(self.qr_size.get())})
        except:
            messagebox.showwarning("Warning","Please key-in scale number")
            return
            
           
        print(f"qr filter  {self.qr_filter.get()}")
        roi.update({"qr_filter":self.qr_filter.get()})
        
        print(f"qr matrix  {self.matrix_var.get()}")
        roi.update({"qr_matrix":self.matrix_var.get()})

        self.start.ActionUi.reload_detection()
        self.start.ActionUi.save_profile()

        self.start.MainUi.roi_setting_frame.refresh_roi_settings()
    
    def refresh(self):
        ret, roi =  self.start.Profile.get_selected_roi()
        if not ret:
            return
            
        #Margin set 0
        roi.update({"margin":0})

        #Add On for sub type
        if roi.get("qr_size") is None:
            roi.update({"qr_size":0})
            print(f"adding qr_size {str(roi.get('qr_size'))}")
            self.start.Main.flag_reload = True
            self.start.Profile.flag_save = True

        #Add On for sub type
        if roi.get("qr_filter") is None:
            roi.update({"qr_filter":""})
            print(f"adding qr_filter {str(roi.get('qr_filter'))}")
            self.start.Main.flag_reload = True
            self.start.Profile.flag_save = True
            
        #Add On for sub type
        if roi.get("qr_matrix") is None:
            roi.update({"qr_matrix":True})
            print(f"adding qr_matrix {str(roi.get('qr_matrix'))}")
            self.start.Main.flag_reload = True
            self.start.Profile.flag_save = True

        self.qr_size.delete(0, END)
        self.qr_size.insert(0, str(roi.get("qr_size")))
        
        self.qr_filter.set(roi.get("qr_filter"))            
            
        self.matrix_var.set(roi.get("qr_matrix"))