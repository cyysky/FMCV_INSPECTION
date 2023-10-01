from tkinter import *
from tkinter import messagebox
from tkinter import ttk
from tkinter import filedialog
from tkinter.messagebox import askokcancel, showinfo, WARNING, showwarning

import traceback

import copy

import numpy
import base64
import cv2

from FMCV import Logging

import time

class ROISettingCobotFrame(ttk.Frame):

    def __init__(self, start, parent, *args, **kwargs):
        #https://stackoverflow.com/questions/17466561/best-way-to-structure-a-tkinter-application/17470842
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.start = start
     
        
        Label(self, text = 'Robot position\n x,y,z,rx,ry,rz(mm)').pack()
        self.position_r_xyz_entry = position_r_xyz_entry = Entry(self)
        position_r_xyz_entry.pack()
        position_r_xyz_entry.bind('<Return>',self.save)
        
        ttk.Label(self, text="Absolute Joint Move").pack()
        self.absolute_joint_var = BooleanVar()
        self.absolute_joint_var.set(True)
        absolute_joint_checkbox = ttk.Checkbutton(self, variable=self.absolute_joint_var, command=lambda: Logging.debug("enable Absolute Joint Move",self.absolute_joint_var.get()))
        absolute_joint_checkbox.pack()
        
        btn_get_tcp = Button(self,text ="Get position")
        btn_get_tcp.pack(side=TOP)
        btn_get_tcp.configure(command=self.get_robot_tcp)

        Label(self, text = 'Move Speed mm/s').pack()
        self.speed_entry = speed_entry = Entry(self)
        speed_entry.pack()
        speed_entry.bind('<Return>',self.save)
        
        Label(self, text = 'Acceration mm/s').pack()
        self.accelerate_entry = accelerate_entry = Entry(self)
        accelerate_entry.pack()
        accelerate_entry.bind('<Return>',self.save)
        
        btn_move = Button(self,text ="Move to position")
        btn_move.pack(side=TOP)
        btn_move.configure(command=self.to_robot_tcp)
       
        btn_save = Button(self,text ="Save")
        btn_save.pack(side=TOP)
        btn_save.configure(command=self.save)
        
        if start.Config.mode_name != "ENGINEER":
            btn_collect["state"] = DISABLED
            box_size.config(state = "disable")
            btn_calculate.config(state = "disable")
    
    def to_robot_tcp(self):
        ret, roi = self.start.Profile.get_selected_roi()
        if ret:
            if roi.get("pos_r_xyz") is not None:
                if roi.get("absolute_joint"):
                    self.start.Com.select_user_coordinate_system(0)
                    self.start.Com.select_tool_coordinate_system(0)
                    self.start.Com.to_tcp([0,0,0,0,0,0])
                    time.sleep(0.1)
                    self.start.pub("com/robot/to_tcp",roi.get("pos_r_xyz"),relative_move=False,speed=roi.get("speed"), accel=roi.get('accelerate'))
                else:
                    self.start.Com.select_user_coordinate_system(10)
                    self.start.Com.select_tool_coordinate_system(10)
                    self.start.Com.to_tcp([0,0,0,0,0,0])
                    time.sleep(0.1)
                    self.start.pub("com/robot/to_tcp",roi.get("pos_r_xyz"),relative_move=False,speed=roi.get("speed"), accel=roi.get('accelerate'))
            
    
    def get_robot_tcp(self):
        ret, roi = self.start.Profile.get_selected_roi()
        if ret:
            if roi.get("absolute_joint"):
                self.start.Com.select_user_coordinate_system(0)
                self.start.Com.select_tool_coordinate_system(0)
                self.start.Com.to_tcp([0,0,0,0,0,0])
                time.sleep(0.1)
                roi["pos_r_xyz"] = self.start.pub("com/robot/get_tcp").get("get_tcp")
                self.refresh()
            else:
                self.start.Com.select_user_coordinate_system(10)
                self.start.Com.select_tool_coordinate_system(10)
                self.start.Com.to_tcp([0,0,0,0,0,0])
                time.sleep(0.1)
                roi["pos_r_xyz"] = self.start.pub("com/robot/get_tcp").get("get_tcp")
                self.refresh()
    
    def save(self):
        if self.start.Users.login_admin():
            ret, roi = self.start.Profile.get_selected_roi()
            if ret:
                #XYZ x,y,z,rx,ry,rz to chessboard to profile
                try:            
                    float_numbers = [float(number) for number in self.position_r_xyz_entry.get().split(',')]
                    Logging.debug("Chessboard Origin offset z,y,z,rx,ry,rz", float_numbers)
                    roi.update({"pos_r_xyz":float_numbers})
                except:
                    traceback.print_exc()
                    messagebox.showwarning( "Warning","Please key-in z,y,z,rx,ry,rz example: 0,0,150,0,0,0 in mm only")
                    
                #Save speed to profile
                try:            
                    print(f"speed {float(self.speed_entry.get())}")
                    roi.update({"speed":float(self.speed_entry.get())})
                except:
                    messagebox.showwarning("Please key-in speed in number", "Warning") 
                    
                #Save speed to profile
                try:            
                    print(f"accelerate {float(self.accelerate_entry.get())}")
                    roi.update({"accelerate":float(self.accelerate_entry.get())})
                except:
                    messagebox.showwarning("Please key-in accelerate in number", "Warning")
                    
                roi.update({"absolute_joint":self.absolute_joint_var.get()})
                
                self.start.ActionUi.reload_detection()
                self.start.ActionUi.save_profile()
            else:
                messagebox.showwarning("Warning","Please select roi on left")
            self.start.MainUi.roi_setting_frame.refresh_roi_settings()
        
    def refresh(self):
        ret, roi = self.start.Profile.get_selected_roi()
        if ret:
            if roi.get("pos_r_xyz") is None:
                roi.update({"pos_r_xyz":[0.0,0,0,0.0,0.0,0.0]})
                print(f"adding position xyz rxryrz {str(roi.get('pos_r_xyz'))}")
                self.start.Main.flag_reload = True
                self.start.Profile.flag_save = True
                
            if roi.get("speed") is None:
                roi.update({"speed":20})
                print(f"adding speed {str(roi.get('speed'))}")
                self.start.Main.flag_reload = True
                self.start.Profile.flag_save = True
                
            if roi.get("accelerate") is None:
                roi.update({"accelerate":100})
                print(f"adding accelerate {str(roi.get('accelerate'))}")
                self.start.Main.flag_reload = True
                self.start.Profile.flag_save = True
            
            if roi.get("absolute_joint") is None:
                roi.update({"absolute_joint":True})
                self.start.Main.flag_reload = True
                self.start.Profile.flag_save = True
             
            self.absolute_joint_var.set(roi.get("absolute_joint"))
                
            self.position_r_xyz_entry.delete(0, END)
            self.position_r_xyz_entry.insert(0, ', '.join([str(round(number, 4)) for number in roi.get("pos_r_xyz")]))
            
            self.speed_entry.delete(0, END)
            self.speed_entry.insert(0, str(roi.get("speed")))
            
            self.accelerate_entry.delete(0, END)
            self.accelerate_entry.insert(0, str(roi.get("accelerate")))