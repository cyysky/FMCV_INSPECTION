from tkinter import *
from tkinter import messagebox
from tkinter import ttk
import copy
import traceback

import webbrowser
from pathlib import Path

from FMCV.Cv import Cv

class ROISettingANOFrame(ttk.Frame):

    def __init__(self, start, parent, *args, **kwargs):
        #https://stackoverflow.com/questions/17466561/best-way-to-structure-a-tkinter-application/17470842
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.start = start
        
        #ANO Frame
        ano_frame = self
        
        btn_open_folder = Button(ano_frame,text ="Open Folder", command = lambda : webbrowser.open(start.ANO.DATASET_PATH))
        btn_open_folder.pack(side=TOP)

        
        btn_save_roi = Button(ano_frame,text ="Save ROI Image To Folder")
        btn_save_roi.pack(side=TOP)
        btn_save_roi.configure(command=self.save_roi_image)   
        
        Label(ano_frame, text = 'Thresh').pack()
        self.ai_minimal = ai_minimal = Entry(ano_frame)
        ai_minimal.pack()
        
        Label(ano_frame, text = 'Train ANO').pack()
        train_btn = Button(ano_frame, text ="Train", command = start.ActionUi.ano_train)
        train_btn.pack(side=TOP)
        
        ai_minimal.bind('<Return>',self.ai_minimal_entry_handler)
        
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
            btn_save_roi["state"] = DISABLED
            train_btn["state"] = DISABLED
            ai_minimal["state"] = DISABLED
            btn_save["state"] = DISABLED

    def ai_minimal_entry_handler(*args):
        print(args)
        self.ai_minimal.select_range(0, 'end')
        

    def save_roi_image(self):
        if not self.start.Users.login_user():
            return
            
        ret, roi = self.start.Profile.get_selected_roi()
        if not ret:
            return 
        
        if self.start.ViewEvent.live and not self.start.ViewEvent.source:
            ret, rotated_frame = self.start.Profile.get_selected_roi_frame()
        else:
            ret, rotated_frame = self.start.MainUi.view.get_image()
           
        if not ret:
            return

        x1 = roi['x1'] 
        y1 = roi['y1'] 
        x2 = roi['x2'] 
        y2 = roi['y2']
        
        #frame = list(self.start.Cam.get_image().values())[self.start.MainUi.cam_pos]
        #rotated_frame = Cv.get_rotate(roi['rotate'],frame)
        pth = self.start.Profile.write_ano_image(copy.deepcopy(rotated_frame[y1:y2,x1:x2]))
        print(pth)
        self.start.MainUi.write(f'ROI Image Saved to {pth}')
            
    def save(self):
        if self.start.Users.login_admin():
            roi_index = self.start.MainUi.roi_index
            if  roi_index > -1:            
                #Save ai_minimal to profile
                try:            
                    print(f"AI ANO minimal {float(self.ai_minimal.get())}")
                    self.start.Profile.loaded_profile[self.start.MainUi.cam_pos][self.start.MainUi.cmb_pos]["roi"][roi_index].update({"minimal":float(self.ai_minimal.get())})
                except:
                    traceback.print_exc()
                    messagebox.showwarning("Please key-in minimal decimal only", "Warning")
                
                #Save blur to profile
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
                        
                self.start.ActionUi.reload_detection()
                self.start.ActionUi.save_profile()
            else:
                messagebox.showinfo("Info","Please select roi on left")
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
                
            self.blur.delete(0, END)
            self.blur.insert(0, str(roi.get("blur")))
           
            self.ai_minimal.delete(0, END)
            self.ai_minimal.insert(0, str(roi.get("minimal")))