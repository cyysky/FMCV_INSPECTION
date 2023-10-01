from tkinter import *
from tkinter import messagebox
from tkinter import ttk
from tkinter import filedialog

import copy

import numpy
import base64
import cv2

from FMCV.Cv import Cv, Filter

class ROISettingLocateFrame(ttk.Frame):

    def __init__(self, start, parent, *args, **kwargs):
        #https://stackoverflow.com/questions/17466561/best-way-to-structure-a-tkinter-application/17470842
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.start = start
        
        #Label(self, text = 'Search margin pixel').pack()
        #self.margin_entry = margin = Entry(self)
        #margin.pack()
        #margin.bind('<Return>',self.save)
        
        Label(self, text = 'Blur').pack()
        self.blur = blur = Entry(self)
        blur.pack()
        blur.bind('<Return>',self.save)
        
        btn_save = Button(self,text ="Save")
        btn_save.pack(side=TOP)
        btn_save.configure(command=self.save)  
        
        if start.Config.mode_name != "ENGINEER":
            btn_save["state"] = DISABLED
        
    
    def save(self):
        if not self.start.Users.login_admin():
            return
        
        ret, roi =  self.start.Profile.get_selected_roi()
        if not ret:
            messagebox.showwarning("Warning","Please select roi on left")
            return
        
        #Save blur to profile
        try:            
            print(f"blur {float(self.blur.get())}")
            roi.update({"blur":int(self.blur.get())})
        except:
            messagebox.showwarning("Warning","Please key-in blur number")
        
        #Save margin to profile
        #try:            
        #    print(f"margin {int(self.margin_entry.get())}")
        #    roi.update({"margin":int(self.margin_entry.get())})
        #except:
        #    messagebox.showwarning("Warning","Please key-in margin integer only")
                        
                        
        self.start.ActionUi.reload_detection()
        self.start.ActionUi.save_profile()

        self.start.MainUi.roi_setting_frame.refresh_roi_settings()
    
    def refresh(self):
        ret, roi =  self.start.Profile.get_selected_roi()
        if not ret:
            return
            
        #self.margin_entry.delete(0, END)
        #self.margin_entry.insert(0, str(roi.get("margin")))
        
        #Add On for sub type
        if roi.get("blur") is None:
            roi.update({"blur":0})
            print(f"adding blur {str(roi.get('blur'))}")
            self.start.Main.flag_reload = True
            self.start.Profile.flag_save = True
            
        self.blur.delete(0, END)
        self.blur.insert(0, str(roi.get("blur")))
            