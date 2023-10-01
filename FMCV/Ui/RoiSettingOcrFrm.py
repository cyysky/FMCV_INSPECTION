from tkinter import *
from tkinter import messagebox
from tkinter import ttk
from tkinter import filedialog

import copy

import numpy
import base64
import cv2

class ROISettingOcrFrame(ttk.Frame):

    def __init__(self, start, parent, *args, **kwargs):
        #https://stackoverflow.com/questions/17466561/best-way-to-structure-a-tkinter-application/17470842
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.start = start
        
        Label(self, text = 'Regular Expression Match').pack()
        self.regular_expression = regular_expression = Entry(self)
        regular_expression.pack()
        regular_expression.bind('<Return>',self.save)
        
        btn_save = Button(self,text ="Save")
        btn_save.pack(side=TOP)
        btn_save.configure(command=self.save)  
        
        Label(self, text = 'Search margin pixel').pack()
        self.margin_entry = margin = Entry(self)
        margin.pack()
        margin.bind('<Return>',self.save)
        
        Label(self, text = 'Blur').pack()
        self.blur = blur = Entry(self)
        blur.pack()
        blur.bind('<Return>',self.save)
        
        if start.Config.mode_name != "ENGINEER":
            btn_save["state"] = DISABLED
    
    def save(self):
        if self.start.Users.login_admin():
            roi_index = self.start.MainUi.roi_index
            if  roi_index > -1:            

                try:            
                    print(f"blur {float(self.blur.get())}")
                    self.start.Profile.loaded_profile[self.start.MainUi.cam_pos][self.start.MainUi.cmb_pos]["roi"][roi_index].update({"blur":int(self.blur.get())})
                except:
                    messagebox.showwarning("Please key-in blur number", "Warning")
                
                #Save margin to profile
                try:            
                    print(f"margin {int(self.margin_entry.get())}")
                    self.start.Profile.loaded_profile[self.start.MainUi.cam_pos][self.start.MainUi.cmb_pos]["roi"][roi_index].update({"margin":int(self.margin_entry.get())})
                except:
                    messagebox.showwarning("Please key-in margin integer only", "Warning")
                
                try:            
                    print(f"re {self.regular_expression.get()}")
                    self.start.Profile.loaded_profile[self.start.MainUi.cam_pos][self.start.MainUi.cmb_pos]["roi"][roi_index].update({"re":self.regular_expression.get()})
                except:
                    messagebox.showwarning("Please key-in regular expression", "Warning")
                
                self.start.ActionUi.reload_detection()
                self.start.ActionUi.save_profile()
            else:
                messagebox.showwarning("Please select roi on left", "Warning")
            self.start.MainUi.roi_setting_frame.refresh_roi_settings()
    
    def refresh(self):
        if self.start.MainUi.roi_index > -1 :
            
            roi = self.start.Profile.loaded_profile[self.start.MainUi.cam_pos][self.start.MainUi.cmb_pos]["roi"][self.start.MainUi.roi_index]
            
            self.margin_entry.delete(0, END)
            self.margin_entry.insert(0, str(roi.get("margin")))
            
            #Add On for sub type
            if roi.get("blur") is None:
                roi.update({"blur":30})
                print(f"adding blur {str(roi.get('blur'))}")
                self.start.Main.flag_reload = True
                self.start.Profile.flag_save = True
                
            #Add On regular expression search
            if roi.get("re") is None:
                roi.update({"re":r"[A-Za-z]\s*[0-9_]\s*[\s]"})
                print(f"adding regular expression re {str(roi.get('re'))}")
                self.start.Main.flag_reload = True
                self.start.Profile.flag_save = True
                
            self.regular_expression.delete(0, END)
            self.regular_expression.insert(0, str(roi.get("re")))
                
            self.blur.delete(0, END)
            self.blur.insert(0, str(roi.get("blur")))
            
            