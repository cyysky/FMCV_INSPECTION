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

class ROISetting3dCalibrationFrame(ttk.Frame):

    def __init__(self, start, parent, *args, **kwargs):
        #https://stackoverflow.com/questions/17466561/best-way-to-structure-a-tkinter-application/17470842
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.start = start
     
        Label(self, text = 'Data Collection').pack()
        btn_collect = Button(self,text ="Collect")
        btn_collect.pack(side=TOP)
        btn_collect.configure(command=lambda:self.start.pub("tool/calibrate3d/get_data"))  
        
        Label(self, text = 'Box size(mm)').pack()
        self.box_size = box_size = Entry(self)
        box_size.pack()
        box_size.bind('<Return>',self.save)
        
        Label(self, text = 'Chessboard robot origin\n x,y,z,rx,ry,rz(mm)').pack()
        self.offset_r_xyz_entry = offset_r_xyz_entry = Entry(self)
        offset_r_xyz_entry.pack()
        offset_r_xyz_entry.bind('<Return>',self.save)
        
        btn_move = Button(self,text ="Move to chessboard")
        btn_move.pack(side=TOP)
        btn_move.configure(command=lambda:self.start.pub("tool/calibrate3d/move_to_chessboard"))
        
        btn_move = Button(self,text ="Set user coordinate")
        btn_move.pack(side=TOP)
        btn_move.configure(command=lambda:self.start.pub("tool/calibrate3d/set_user_coordinate"))
        
        btn_user_frame = Button(self,text ="Reset user coordinate")
        btn_user_frame.pack(side=TOP)
        btn_user_frame.configure(command=lambda:self.start.pub("tool/calibrate3d/reset_user_coordinate"))
        
        Label(self, text = 'Boxes Rows,Columns').pack()
        self.box_row_col_entry = box_row_col_entry = Entry(self)
        box_row_col_entry.pack()
        box_row_col_entry.bind('<Return>',self.save)
        
        btn_calculate = Button(self,text ="Calibrate")
        btn_calculate.pack(side=TOP)
        btn_calculate.configure(command=lambda:self.start.pub("tool/calibrate3d/calibrate"))  
        
        btn_save = Button(self,text ="Save")
        btn_save.pack(side=TOP)
        btn_save.configure(command=self.save)
        
        if start.Config.mode_name != "ENGINEER":
            btn_collect["state"] = DISABLED
            box_size.config(state = "disable")
            btn_calculate.config(state = "disable")
            
    
    def save(self):
        if self.start.Users.login_admin():
            roi_index = self.start.MainUi.roi_index
            if  roi_index > -1:
                #Chessboard size in mm to profile
                try:            
                    print(f"box size {float(self.box_size.get())}")
                    self.start.Profile.loaded_profile[self.start.MainUi.cam_pos][self.start.MainUi.cmb_pos]["roi"][roi_index].update({"box_size":float(self.box_size.get())})
                except:
                    traceback.print_exc()
                    messagebox.showwarning( "Warning","Please key-in box size in mm only")
                
                #Traget offset x,y,z,rx,ry,rz to chessboard to profile
                try:            
                    float_numbers = [float(number) for number in self.offset_r_xyz_entry.get().split(',')]
                    Logging.debug("Chessboard Origin offset z,y,z,rx,ry,rz", float_numbers)
                    self.start.Profile.loaded_profile[self.start.MainUi.cam_pos][self.start.MainUi.cmb_pos]["roi"][roi_index].update({"offset_r_xyz":float_numbers})
                except:
                    traceback.print_exc()
                    messagebox.showwarning( "Warning","Please key-in z,y,z,rx,ry,rz example: 0,0,150,0,0,0 in mm only")
                    
                #chessboard row and column to profile
                try:            
                    float_numbers = [int(number)-1 for number in self.box_row_col_entry.get().split(',')]
                    Logging.debug("ROWS,COLUMNS", float_numbers)
                    self.start.Profile.loaded_profile[self.start.MainUi.cam_pos][self.start.MainUi.cmb_pos]["roi"][roi_index].update({"box_rows_cols":float_numbers})
                except:
                    traceback.print_exc()
                    messagebox.showwarning( "Warning","Please key-in Chessboard rows,columns example: 9,6 only")
                    
                self.start.ActionUi.reload_detection()
                self.start.ActionUi.save_profile()
            else:
                messagebox.showwarning("Warning","Please select roi on left")
            self.start.MainUi.roi_setting_frame.refresh_roi_settings()
        
    def refresh(self):
        if self.start.MainUi.roi_index > -1 :
            
            roi = self.start.Profile.loaded_profile[self.start.MainUi.cam_pos][self.start.MainUi.cmb_pos]["roi"][self.start.MainUi.roi_index]
            
            #Add On for sub type
            if roi.get("box_size") is None:
                roi.update({"box_size":30})
                print(f"adding box_size {str(roi.get('box_size'))}")
                self.start.Main.flag_reload = True
                self.start.Profile.flag_save = True
                
            if roi.get("box_rows_cols") is None:
                roi.update({"box_rows_cols":[8,5]})
                self.start.Main.flag_reload = True
                self.start.Profile.flag_save = True
                
            if roi.get("offset_r_xyz") is None:
                roi.update({"offset_r_xyz":[0.0,0.0,150.0,0.0,0.0,0.0]})
                print(f"adding offset xyz rxryrz {str(roi.get('offset_r_xyz'))}")
                self.start.Main.flag_reload = True
                self.start.Profile.flag_save = True
                
                
                
            self.box_size.delete(0, END)
            self.box_size.insert(0, str(roi.get("box_size")))
            
            self.offset_r_xyz_entry.delete(0, END)
            self.offset_r_xyz_entry.insert(0, ', '.join([str(number) for number in roi.get("offset_r_xyz")]))
            
            self.box_row_col_entry.delete(0, END)
            self.box_row_col_entry.insert(0, ', '.join([str(number+1) for number in roi.get("box_rows_cols")]))
