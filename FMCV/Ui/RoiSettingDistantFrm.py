from tkinter import *
from tkinter import messagebox
from tkinter import ttk
from tkinter import filedialog

import copy

import numpy
import base64
import cv2

from FMCV.Cv import Cv, Filter
from FMCV import Logging

class ROISettingDistantFrame(ttk.Frame):

    def __init__(self, start, parent, *args, **kwargs):
        #https://stackoverflow.com/questions/17466561/best-way-to-structure-a-tkinter-application/17470842
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.start = start
        
        
        Label(self, text = 'Refer to ROI').pack()
        self.refer = Entry(self)
        self.refer.pack()
        self.refer.bind('<Return>',self.save)
        
        Label(self, text = 'Search margin pixel').pack()
        self.margin_entry = Entry(self)
        self.margin_entry.pack()
        self.margin_entry.bind('<Return>',self.save)
        
        Label(self, text = 'Minimal Score').pack()
        self.minimal_entry = Entry(self)
        self.minimal_entry.pack()
        self.minimal_entry.bind('<Return>',self.save)
        
        Label(self, text = 'Method').pack()
        self.method = Entry(self)
        self.method.pack()
        self.method.bind('<Return>',self.save)
        
        Label(self, text = 'Blur').pack()
        self.blur = Entry(self)
        self.blur.pack()
        self.blur.bind('<Return>',self.save)

        Label(self, text = 'MM per Pixel').pack()
        self.mm_pixel = Entry(self)
        self.mm_pixel.pack()
        self.mm_pixel.bind('<Return>',self.save)
        
        Label(self, text = 'Distance Torelance').pack()
        Label(self, text = 'Min (mm)').pack()
        self.distance_min = Entry(self)
        self.distance_min.pack()
        self.distance_min.bind('<Return>',self.save)
        Label(self, text = 'Max (mm)').pack()
        self.distance_max = Entry(self)
        self.distance_max.pack()
        self.distance_max.bind('<Return>',self.save)        
        Label(self, text = 'X Min (mm)').pack()
        self.distance_x_min = Entry(self)
        self.distance_x_min.pack()
        self.distance_x_min.bind('<Return>',self.save)
        Label(self, text = 'X Max (mm)').pack()
        self.distance_x_max = Entry(self)
        self.distance_x_max.pack()
        self.distance_x_max.bind('<Return>',self.save)
        Label(self, text = 'Y Min (mm)').pack()
        self.distance_y_min = Entry(self)
        self.distance_y_min.pack()
        self.distance_y_min.bind('<Return>',self.save)
        Label(self, text = 'Y Max (mm)').pack()
        self.distance_y_max = Entry(self)
        self.distance_y_max.pack()
        self.distance_y_max.bind('<Return>',self.save)
     
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
            messagebox.showwarning("Warning","Please select roi on left" )
            return
            
        #Save mm_pixel to profile
        try:            
            Logging.info(f"mm_pixel {float(self.mm_pixel.get())}")
            roi.update({"mm_pixel":float(self.mm_pixel.get())})
        except:
            messagebox.showwarning("Warning","Please key-in mm per pixel number")
            
        #Save margin to profile
        try:            
            Logging.info(f"margin {int(self.margin_entry.get())}")
            roi.update({"margin":int(self.margin_entry.get())})
        except:
            messagebox.showwarning("Warning","Please key-in margin integer only")
            
        #Save minimal_entry to profile
        try:            
            Logging.info(f"minimal {float(self.minimal_entry.get())}")
            roi.update({"minimal":float(self.minimal_entry.get())})
        except:
            messagebox.showwarning("Warning","Please key-in minimal decimal only 0.0 - 1.0")
        #Save blur to profile
        try:            
            Logging.info(f"blur {int(self.blur.get())}")
            roi.update({"blur":int(self.blur.get())})
        except:
            messagebox.showwarning("Warning","Please key-in blur number")
            
        #Save method to profile
        try:            
            Logging.info(f"method {int(self.method.get())}")
            roi.update({"method":int(self.method.get())})
        except:
            messagebox.showwarning("Warning","Please key-in method number")   

        #Save method to profile           
        Logging.info(f"refer {self.refer.get()}")
        roi.update({"refer":self.refer.get()})         


         #Save torelance to profile
        try:            
            Logging.info(f"distance_min {float(self.distance_min.get())}")
            roi.update({"distance_min":float(self.distance_min.get())})
        except:
            messagebox.showwarning("Warning","Please key-in torelance min in number")
        try:            
            Logging.info(f"distance_max {float(self.distance_max.get())}")
            roi.update({"distance_max":float(self.distance_max.get())})
        except:
            messagebox.showwarning("Warning","Please key-in torelance max in number")
            
        #Save x_torelance to profile
        try:            
            Logging.info(f"distance_x_min {float(self.distance_x_min.get())}")
            roi.update({"distance_x_min":float(self.distance_x_min.get())})
        except:
            messagebox.showwarning("Warning","Please key-in X torelance min in number")
        try:            
            Logging.info(f"distance_x_max {float(self.distance_x_max.get())}")
            roi.update({"distance_x_max":float(self.distance_x_max.get())})
        except:
            messagebox.showwarning("Warning","Please key-in X torelance max in number")
            
         #Save y_torelance to profile
        try:            
            Logging.info(f"distance_y_min {float(self.distance_y_min.get())}")
            roi.update({"distance_y_min":float(self.distance_y_min.get())})
        except:
            messagebox.showwarning("Warning","Please key-in Y torelance min in number")
        try:            
            Logging.info(f"distance_y_max {float(self.distance_y_max.get())}")
            roi.update({"distance_y_max":float(self.distance_y_max.get())})
        except:
            messagebox.showwarning("Warning","Please key-in Y torelance max in number")

        self.start.ActionUi.reload_detection()
        self.start.ActionUi.save_profile()

        self.start.MainUi.roi_setting_frame.refresh_roi_settings()
    
    def refresh(self):
        ret, roi =  self.start.Profile.get_selected_roi()
        if not ret:
            return
            
        if roi.get("mask") is None:
            img = roi.get('img')
            if img is not None:
                image_height = img.shape[0]
                image_width = img.shape[1]
                number_of_color_channels = 1
                color = (255)
                pixel_array = numpy.full((image_height, image_width, number_of_color_channels), color, dtype=numpy.uint8)
                
                roi.update({"mask_64":base64.b64encode(cv2.imencode('.png',copy.deepcopy(pixel_array))[1]).decode()})
                roi.update({'mask':copy.deepcopy(pixel_array)})
                self.start.Main.flag_reload = True
                self.start.Profile.flag_save = True
            
        #Add On for sub type
        if roi.get("mm_pixel") is None:
            roi.update({"mm_pixel":1.0})
            Logging.info(f"adding mm_pixel {str(roi.get('mm_pixel'))}")
            self.start.Main.flag_reload = True
            self.start.Profile.flag_save = True
            
        if roi.get("blur") is None:
            roi.update({"blur":30})
            Logging.info(f"adding blur {str(roi.get('blur'))}")
            self.start.Main.flag_reload = True
            self.start.Profile.flag_save = True
            
        if roi.get("method") is None:
            roi.update({"method":5})
            Logging.info(f"adding method {str(roi.get('method'))}")
            self.start.Main.flag_reload = True
            self.start.Profile.flag_save = True
            
        if roi.get("refer") is None:
            roi.update({"refer":""})
            Logging.info(f"adding refer {str(roi.get('refer'))}")
            self.start.Main.flag_reload = True
            self.start.Profile.flag_save = True
            
        if roi.get("distance_min") is None:
            roi.update({"distance_min":0.0})
            Logging.info(f"adding distance_min {str(roi.get('distance_min'))}")
            self.start.Main.flag_reload = True
            self.start.Profile.flag_save = True
        if roi.get("distance_max") is None:
            roi.update({"distance_max":0.0})
            Logging.info(f"adding distance_max {str(roi.get('distance_max'))}")
            self.start.Main.flag_reload = True
            self.start.Profile.flag_save = True
        if roi.get("distance_y_min") is None:
            roi.update({"distance_y_min":0.0})
            Logging.info(f"adding distance_y_min {str(roi.get('distance_y_min'))}")
            self.start.Main.flag_reload = True
            self.start.Profile.flag_save = True
        if roi.get("distance_y_max") is None:
            roi.update({"distance_y_max":0.0})
            Logging.info(f"adding distance_y_max {str(roi.get('distance_y_max'))}")
            self.start.Main.flag_reload = True
            self.start.Profile.flag_save = True
        if roi.get("distance_x_min") is None:
            roi.update({"distance_x_min":0.0})
            Logging.info(f"adding distance_x_min {str(roi.get('distance_x_min'))}")
            self.start.Main.flag_reload = True
            self.start.Profile.flag_save = True
        if roi.get("distance_x_max") is None:
            roi.update({"distance_x_max":0.0})
            Logging.info(f"adding distance_x_max {str(roi.get('distance_x_max'))}")
            self.start.Main.flag_reload = True
            self.start.Profile.flag_save = True

            
        self.margin_entry.delete(0, END)
        self.margin_entry.insert(0, str(roi.get("margin")))
        
        self.minimal_entry.delete(0, END)
        self.minimal_entry.insert(0, str(roi.get("minimal")))
        
        self.mm_pixel.delete(0, END)
        self.mm_pixel.insert(0, str(roi.get("mm_pixel")))
        
        self.blur.delete(0, END)
        self.blur.insert(0, str(roi.get("blur")))
        
        self.method.delete(0, END)
        self.method.insert(0, str(roi.get("method")))
        
        self.refer.delete(0, END)
        self.refer.insert(0, str(roi.get("refer")))
        
        
        self.distance_min.delete(0, END)
        self.distance_min.insert(0, str(roi.get("distance_min")))
        self.distance_max.delete(0, END)
        self.distance_max.insert(0, str(roi.get("distance_max")))
        self.distance_x_min.delete(0, END)
        self.distance_x_min.insert(0, str(roi.get("distance_x_min")))
        self.distance_x_max.delete(0, END)
        self.distance_x_max.insert(0, str(roi.get("distance_x_max")))
        self.distance_y_min.delete(0, END)
        self.distance_y_min.insert(0, str(roi.get("distance_y_min")))
        self.distance_y_max.delete(0, END)
        self.distance_y_max.insert(0, str(roi.get("distance_y_max")))