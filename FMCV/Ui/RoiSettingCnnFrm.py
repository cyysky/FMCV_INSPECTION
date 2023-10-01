from tkinter import *
from tkinter import messagebox
from tkinter import ttk
import copy
import traceback
from FMCV.Cv import Cv

class ROISettingCNNFrame(ttk.Frame):

    def __init__(self, start, parent, *args, **kwargs):
        #https://stackoverflow.com/questions/17466561/best-way-to-structure-a-tkinter-application/17470842
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.start = start
        
        #CNN Frame
        cnn_frame = self
        

        Label(cnn_frame, text = 'Images Folders').pack()
        btn_open_folder = Button(cnn_frame,text ="Open Folder", command = lambda : start.ActionUi.open_folder(folder_cmb.get()))
        btn_open_folder.pack(side=TOP)
        
        self.folder_cmb = folder_cmb = ttk.Combobox(cnn_frame)
        folder_cmb['values'] = tuple(start.Profile.get_image_folders_list())
        folder_cmb.pack()

        btn_save_roi = Button(cnn_frame,text ="Save ROI Image To Folder")
        btn_save_roi.pack(side=TOP)
        btn_save_roi.configure(command=self.save_roi_image)   

        Label(cnn_frame, text = 'CNN').pack()
        btn_set_class = Button(cnn_frame,text ="Set target class")
        btn_set_class.pack(side=TOP)
        btn_set_class.configure(command=self.update_roi_class)   
        self.lbl_class = Label(cnn_frame, text = 'Target class:')
        self.lbl_class.pack()
        
        self.ai_minimal = ai_minimal = Entry(cnn_frame)
        ai_minimal.pack()
        
        Label(cnn_frame, text = 'Train CNN').pack()
        train_btn = Button(cnn_frame, text ="Train", command = start.ActionUi.cnn_train)
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
            btn_set_class["state"] = DISABLED
            train_btn["state"] = DISABLED
            ai_minimal["state"] = DISABLED
            btn_save.config(state = "disable")
            
    def ai_minimal_entry_handler(*args):
        print(args)
        self.ai_minimal.select_range(0, 'end')
        
    def update_roi_class(self):
        if self.start.Users.login_admin():
            roi_index = self.start.MainUi.roi_index
            if  roi_index > -1:
                self.start.Profile.loaded_profile[self.start.MainUi.cam_pos][self.start.MainUi.cmb_pos]["roi"][roi_index].update({"class":self.folder_cmb.get()})
                self.start.ActionUi.reload_detection()
                self.start.MainUi.roi_setting_frame.refresh_roi_settings()
            
    def save_roi_image(self):
        if not self.start.Users.login_user():
            print("not user")
            return
            
        ret, roi = self.start.Profile.get_selected_roi()
        if not ret:
            print("no roi")
            return 
            
        if self.start.ViewEvent.live and not self.start.ViewEvent.source:
            ret, rotated_frame = self.start.Profile.get_selected_roi_frame()
            #rotated_frame = Cv.get_rotate(roi['rotate'],frame)
        else:
            ret, rotated_frame = self.start.MainUi.view.get_image()
        if not ret:
            print("no frame")
            return
            
        #self.start.Profile.create_image_folder(self.folder_cmb.get())
        self.folder_cmb['values'] = tuple(self.start.Profile.get_image_folders_list())
        #roi = self.start.Profile.loaded_profile[self.start.MainUi.cam_pos][self.start.MainUi.cmb_pos]["roi"][roi_index]
        x1 = roi['x1'] 
        y1 = roi['y1'] 
        x2 = roi['x2'] 
        y2 = roi['y2']
        #frame = list(self.start.Cam.get_image().values())[self.start.MainUi.cam_pos]
        
        pth = self.start.Profile.write_image(self.folder_cmb.get(), copy.deepcopy(rotated_frame[y1:y2,x1:x2]))
        print(self.folder_cmb.get())
        self.start.MainUi.write(f'ROI Image Saved to {pth}')
            
    def save(self):
        if self.start.Users.login_admin():
            roi_index = self.start.MainUi.roi_index
            if  roi_index > -1:            
                #Save ai_minimal to profile
                try:            
                    print(f"ai minimal {float(self.ai_minimal.get())}")
                    self.start.Profile.loaded_profile[self.start.MainUi.cam_pos][self.start.MainUi.cmb_pos]["roi"][roi_index].update({"minimal":float(self.ai_minimal.get())})
                except:
                    traceback.print_exc()
                    messagebox.showwarning("Warning","Please key-in minimal decimal only 0.0 - 1.0")
                #Save blur to profile
                try:            
                    print(f"blur {float(self.blur.get())}")
                    self.start.Profile.loaded_profile[self.start.MainUi.cam_pos][self.start.MainUi.cmb_pos]["roi"][roi_index].update({"blur":int(self.blur.get())})
                except:
                    messagebox.showwarning("Warning","Please key-in blur number")
                
                #Save margin to profile
                try:            
                    print(f"margin {int(self.margin_entry.get())}")
                    self.start.Profile.loaded_profile[self.start.MainUi.cam_pos][self.start.MainUi.cmb_pos]["roi"][roi_index].update({"margin":int(self.margin_entry.get())})
                except:
                    messagebox.showwarning("Warning","Please key-in margin integer only")
                        
                self.start.ActionUi.reload_detection()
                self.start.ActionUi.save_profile()
            else:
                messagebox.showinfo("Info","Please select roi on left")
            self.start.MainUi.roi_setting_frame.refresh_roi_settings()
        
        
    def refresh(self):
        if self.start.MainUi.roi_index > -1 :
            
            roi = self.start.Profile.loaded_profile[self.start.MainUi.cam_pos][self.start.MainUi.cmb_pos]["roi"][self.start.MainUi.roi_index]
            if roi.get('class') is None:
                roi.update({'class':""})
                
            self.lbl_class.config(text = f"Target Class : {roi.get('class')}")
            
            self.ai_minimal.delete(0, END)
            self.ai_minimal.insert(0, str(roi.get("minimal")))
            
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
            self.folder_cmb['values'] = tuple(self.start.Profile.get_image_folders_list())
        else:
            self.lbl_class.config(text = f"Target class : N/A")