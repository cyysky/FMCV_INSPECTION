from tkinter import *
from tkinter import messagebox
from tkinter import ttk
from tkinter import filedialog
from tkinter.messagebox import askokcancel, showinfo, WARNING, showwarning

import traceback

import copy

import numpy
import numpy as np
import base64
import cv2

from FMCV.Cv import Match
from FMCV import Logging

class ROISettingFiducialFrame(ttk.Frame):

    def __init__(self, start, parent, *args, **kwargs):
        #https://stackoverflow.com/questions/17466561/best-way-to-structure-a-tkinter-application/17470842
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.start = start
        
        
        btn_download_mask = Button(self,text ="Download mask",command=self.download_mask)
        btn_download_mask.pack(side=TOP)
        
        btn_upload_mask = Button(self,text ="Upload mask",command=self.upload_mask)
        btn_upload_mask.pack(side=TOP)
        
        btn_view_mask = Button(self,text ="View mask",command=self.view_mask)
        btn_view_mask.pack(side=TOP)
        
        Label(self, text = 'Auto adjust').pack()
        btn_zero = Button(self,text ="Set roi from result")
        btn_zero.pack(side=TOP)
        btn_zero.configure(command=self.zero)
       
        Label(self, text = 'Fiducial margin pixel').pack()
        self.margin_entry = margin = Entry(self)
        margin.pack()
        margin.bind('<Return>',self.save)
        
        Label(self, text = 'Minimal Score').pack()
        self.minimal_entry = minimal_entry = Entry(self)
        minimal_entry.pack()
        minimal_entry.bind('<Return>',self.save)
        
        Label(self, text = 'MM per Pixel').pack()
        self.mm_pixel = mm_pixel = Entry(self)
        mm_pixel.pack()
        mm_pixel.bind('<Return>',self.save)
        
        Label(self, text = 'Blur').pack()
        self.blur = blur = Entry(self)
        blur.pack()
        blur.bind('<Return>',self.save)
        
        Label(self, text = 'Method').pack()
        self.method = method = Entry(self)
        method.pack()
        method.bind('<Return>',self.save)
        
        ttk.Label(self, text="Enable Angle").pack()
        self.enable_angle_var = BooleanVar()
        self.enable_angle_var.set(True)
        enable_angle_checkbox = ttk.Checkbutton(self, variable=self.enable_angle_var, command=lambda: Logging.debug("enable angle",self.enable_angle_var.get()))
        enable_angle_checkbox.pack()
        
        btn_test = Button(self,text = "Test Angle", command=self.test_angle_2).pack(side=TOP)
        
        # Disabled
        # ttk.Label(self, text="Auto Move").pack()
        self.enable_auto_move_var = BooleanVar()
        # self.enable_auto_move_var.set(True)
        # enable_auto_move = ttk.Checkbutton(self, variable=self.enable_auto_move_var, command=lambda: Logging.debug("enable auto move ",self.enable_auto_move_var.get()))
        # enable_auto_move.pack()
        
        ttk.Label(self, text="Automata").pack()
        btn_calibrate = Button(self,text ="Auto 2D Calibrate")
        btn_calibrate.pack(side=TOP)
        btn_calibrate.configure(command=lambda:self.start.pub("tool/calibrate2d/calibrate"))  

        btn_test = Button(self,text ="Move to Global Center")
        btn_test.pack(side=TOP)
        btn_test.configure(command=lambda:self.start.pub("tool/calibrate2d/test"))
        
        btn_test = Button(self,text ="Set ROI to Global Center")
        btn_test.pack(side=TOP)
        btn_test.configure(command=lambda:self.start.pub("ui/roi/move_roi_center"))
        
        btn_test = Button(self,text ="Move to ROI Center")
        btn_test.pack(side=TOP)
        btn_test.configure(command=lambda:self.start.pub("tool/calibrate2d/test_roi_center"))
        
        btn_test = Button(self,text ="Set User Coordinate")
        btn_test.pack(side=TOP)
        btn_test.configure(command=lambda:self.start.pub("tool/calibrate2d/set_user_coordinate"))
        
        btn_save = Button(self,text ="Save")
        btn_save.pack(side=TOP)
        btn_save.configure(command=self.save)  
        
        
        Label(self, text = 'Remove Calibration').pack()
        btn_test = Button(self,text ="Remove 3D Camera Calibrate")
        btn_test.pack(side=TOP)
        btn_test.configure(command=self.remove_calibrate)  
        
        if start.Config.mode_name != "ENGINEER":
            mm_pixel["state"] = DISABLED
            btn_save.config(state = "disable")
            btn_upload_mask.config(state = "disable")
            btn_zero.config(state = "disable")
  
    
    def test_angle_to_do(self):
        ret, roi = self.start.Profile.get_selected_roi()
        if ret:
            a = Match.match_rotate(roi.get('img'),list(self.start.Cam.get_image().values())[0].copy())
            if a:
                frame, cropped, [x_center, y_center], angle , dst, xy_dst = a

                
                print(dst)
                #fourth_corner = dst[3]
                #dst = numpy.insert(dst, 1, fourth_corner, axis=0)
                #dst = numpy.delete(dst, 4, axis=0)
                #print(dst)
                def interpolate_points(p1, p2, num_points):
                    return np.array([p1 * (1 - t) + p2 * t for t in np.linspace(0, 1, num_points)])

                # Define the four corner points
                top_left = np.array(dst[0])
                top_right = np.array(dst[3])
                bottom_right = np.array(dst[2])
                bottom_left = np.array(dst[1])

                # Chessboard pattern size
                pattern_size = (9, 9)

                # Interpolate points along the top and bottom edges
                top_row = interpolate_points(top_left, top_right, pattern_size[0])
                bottom_row = interpolate_points(bottom_left, bottom_right, pattern_size[0])

                # Interpolate points along the top and bottom edges
                left_col = interpolate_points(top_left, bottom_left, pattern_size[1])
                right_col = interpolate_points(top_right, bottom_right, pattern_size[1])

                # Interpolate points between the top and bottom rows to form a grid
                grid_points = []
                for i in range(pattern_size[1]):
                    col_points = interpolate_points(left_col[i],right_col[i], pattern_size[1])
                    grid_points.append(col_points)

                # Convert the grid points to a NumPy array with shape (81, 1, 2)
                corners = np.array(grid_points).reshape(-1, 1, 2)
                
                # Draw the chessboard pattern on image
                for pt in corners:
                    x, y = pt.ravel().astype(int)
                    frame = cv2.circle(frame, (x, y), 3, (0, 0, 128), -1)
                self.start.MainUi.update_result_frame(frame)
                
                
                
                if roi.get('R') is not None:
                    self.start.Com.select_user_coordinate_system(0) # World Frame selected
                    self.start.Com.select_tool_coordinate_system(0) # End Flage selected 
                    tcp = self.start.Calibrate3d.get_tcp()
                    if tcp is not None:
                        print(self.start.Calibrate3d.average_distance_between_points(corners,[9,9]))
                        distant = np.linalg.norm(np.array([roi['x1'],roi['y1']])-np.array([roi['x2'],roi['y1']]))
                        print(distant)
                        print(roi['x2']-roi['x1'])
                        print(roi['pixel_to_mm'])
                        print((roi['x2']-roi['x1'])*roi['pixel_to_mm'])
                        print(((roi['x2']-roi['x1'])*roi['pixel_to_mm'])/8)
                        box_size = ((roi['x2']-roi['x1'])*roi['pixel_to_mm'])/8
                        print(box_size)
                        output_pos, img_chessboard, pixel_to_mm = self.start.Calibrate3d.calculate(roi.get('R'), roi.get('T'), roi.get('K'), roi.get('D'),
                                                                tcp, offset_r_xyz = roi.get('offset_r_xyz'),
                                                                box_rows_cols = [9,9], box_size = box_size, points = corners)
                                                                
                                                                
                        print(output_pos)
                        eye_to_hand_R = numpy.array(roi.get('R'))
                        eye_to_hand_t = numpy.array(roi.get('T'))
                        user_coordinate_offset =  self.start.Calibrate3d.getCameraPoseGivenHandPose(output_pos, eye_to_hand_R, eye_to_hand_t)
                        
                        print(user_coordinate_offset)
                        self.start.Com.set_user_coordinate_system(user_coordinate_offset,10,"FMCV Board")

    def test_angle(self):
        ret, roi = self.start.Profile.get_selected_roi()
        if ret:
            a = Match.match_rotate(roi.get('img'),list(self.start.Cam.get_image().values())[0].copy())
            if a:
                frame, cropped, [x_center, y_center], angle , dst, xy_dst = a
                self.start.MainUi.update_result_frame(frame)
                
    def test_angle_2(self):
        ret, roi = self.start.Profile.get_selected_roi()
        if not ret:
            return
        
        print("obj roi",roi.get('T'))
        
        a = Match.match_rotate(roi.get('img'),list(self.start.Cam.get_image().values())[0].copy())
        if not a:
            return
            
        frame, cropped, [x_center, y_center], angle , dst, xy_dst = a
        self.start.MainUi.update_result_frame(frame)
        print("dst",dst)
        
        # Define the four corner points
        # Reordering
        reordered_dst = dst[[0, 3, 1, 2]]
        
        if roi.get('R') is None:
            return
            
        self.start.Com.select_user_coordinate_system(0) # World Frame selected
        self.start.Com.select_tool_coordinate_system(0) # End Flage selected 
        tcp = self.start.Calibrate3d.get_tcp()
        
        if tcp is None:
            return
            
        box_size = (roi['x2']-roi['x1'])*roi['pixel_to_mm']
        output_pos, img_chessboard, pixel_to_mm = self.start.Calibrate3d.calculate(roi.get('R'), roi.get('T'), roi.get('K'), roi.get('D'),
                                                                tcp, offset_r_xyz = roi.get('offset_r_xyz'),
                                                                box_rows_cols = [2,2], box_size = box_size, points = reordered_dst)
                
    def remove_calibrate(self):
        if self.start.Users.login_admin():
            ret, roi = self.start.Profile.get_selected_roi()
            if ret:
                roi.pop('R')
                roi.pop('T')
                roi.pop('K')
                roi.pop('D')
                roi.pop('W')
                roi.pop('Q')
                self.start.ActionUi.reload_detection()
                self.start.Profile.flag_save = True
                
    def zero(self): 
        if self.start.Users.login_admin():
            roi_index = self.start.MainUi.roi_index
            if  roi_index > -1:
                try:
                    roi = self.start.Main.results[self.start.MainUi.cam_pos][self.start.MainUi.cmb_pos][roi_index]
           
                    off_x = 0
                    off_y = 0
                    
                    if roi.get("offset_x") is not None:
                        off_x = roi["offset_x"]
                        
                    if roi.get("offset_y") is not None:
                        off_y = roi["offset_y"]
                        
                    roi = self.start.Profile.loaded_profile[self.start.MainUi.cam_pos][self.start.MainUi.cmb_pos]["roi"][roi_index]

                    roi['x1'] = roi['x1'] + off_x
                    roi['y1'] = roi['y1'] + off_y
                    roi['x2'] = roi['x2'] + off_x
                    roi['y2'] = roi['y2'] + off_y
                    self.start.MainUi_Refresh.refresh_edit_roi_rectangle()
                    self.start.Main.flag_reload = True
                    self.start.Profile.flag_save = True
                except:
                    traceback.print_exc()
                    print("Please detect before use this zero feedback function")
                    messagebox.showwarning("Set roi position from result", "Please detect before use this set roi from result function")
                
    def download_mask(self):
        roi_index = self.start.MainUi.roi_index
        if  roi_index > -1:
            roi = self.start.Profile.loaded_profile[self.start.MainUi.cam_pos][self.start.MainUi.cmb_pos]["roi"][roi_index]
            img = roi['img']
            image_height = img.shape[0]
            image_width = img.shape[1]
            number_of_color_channels = 1
            color = (255)
            pixel_array = numpy.full((image_height, image_width, number_of_color_channels), color, dtype=numpy.uint8)

            f = filedialog.asksaveasfilename(initialfile = 'mask.bmp',defaultextension=".bmp",filetypes=[("All Files","*.*"),("bitmap","*.bmp"),("PNG","*.png")])
            print(f)            
            cv2.imwrite(f, pixel_array)
    
    def upload_mask(self):
        if self.start.Users.login_admin():
            roi_index = self.start.MainUi.roi_index
            if  roi_index > -1:
                roi = self.start.Profile.loaded_profile[self.start.MainUi.cam_pos][self.start.MainUi.cmb_pos]["roi"][roi_index]
                file_path = filedialog.askopenfilename(filetypes=(("Image files", "*.jpg;*.bmp;*.png"),("All files", "*.*") ))
                mask = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
                th, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                print(mask.dtype, mask.shape)
                if mask is not None:
                    roi.update({"mask_64":base64.b64encode(cv2.imencode('.png',copy.deepcopy(mask))[1]).decode()})
                    roi.update({'mask':copy.deepcopy(mask)})
                    self.start.MainUi.roi_setting_frame.image_view.set_image(roi['mask'])
                    self.start.Main.flag_reload = True
                    self.start.Profile.flag_save = True
            
             
    def view_mask(self):
        roi_index = self.start.MainUi.roi_index
        if roi_index > -1:
            roi = self.start.Profile.loaded_profile[self.start.MainUi.cam_pos][self.start.MainUi.cmb_pos]["roi"][roi_index]
            masked = cv2.bitwise_and(roi['img'], roi['img'], mask=roi['mask'])
            print(masked.shape)
            self.start.MainUi.roi_setting_frame.image_view.set_image(masked)
           
    
    
    def save(self):
        if self.start.Users.login_admin():
            roi_index = self.start.MainUi.roi_index
            if  roi_index > -1:
                #Save margin to profile
                try:            
                    print(f"margin {int(self.margin_entry.get())}")
                    self.start.Profile.loaded_profile[self.start.MainUi.cam_pos][self.start.MainUi.cmb_pos]["roi"][roi_index].update({"margin":int(self.margin_entry.get())})
                except:
                    messagebox.showwarning("Warning","Please key-in margin integer only")
                    
                #Save minimal_entry to profile
                try:            
                    print(f"minimal {float(self.minimal_entry.get())}")
                    self.start.Profile.loaded_profile[self.start.MainUi.cam_pos][self.start.MainUi.cmb_pos]["roi"][roi_index].update({"minimal":float(self.minimal_entry.get())})
                except:
                    messagebox.showwarning("Warning","Please key-in minimal decimal only 0.0 - 1.0")
                
                #Save mm_pixel to profile
                try:            
                    print(f"mm_pixel {float(self.mm_pixel.get())}")
                    self.start.Profile.loaded_profile[self.start.MainUi.cam_pos][self.start.MainUi.cmb_pos]["roi"][roi_index].update({"mm_pixel":float(self.mm_pixel.get())})
                except:
                    messagebox.showwarning("Warning","Please key-in mm per pixel number")
                    
                #Save blur to profile
                try:            
                    print(f"blur {float(self.blur.get())}")
                    self.start.Profile.loaded_profile[self.start.MainUi.cam_pos][self.start.MainUi.cmb_pos]["roi"][roi_index].update({"blur":int(self.blur.get())})
                except:
                    messagebox.showwarning("Warning","Please key-in blur number")
                    
                #Save method to profile
                try:            
                    print(f"method {float(self.method.get())}")
                    self.start.Profile.loaded_profile[self.start.MainUi.cam_pos][self.start.MainUi.cmb_pos]["roi"][roi_index].update({"method":int(self.method.get())})
                except:
                    messagebox.showwarning("Warning","Please key-in method number")
                
                self.start.Profile.loaded_profile[self.start.MainUi.cam_pos][self.start.MainUi.cmb_pos]["roi"][roi_index].update({"angle_enable":self.enable_angle_var.get()})
                
                self.start.Profile.loaded_profile[self.start.MainUi.cam_pos][self.start.MainUi.cmb_pos]["roi"][roi_index].update({"auto_move":self.enable_auto_move_var.get()})
                
                self.start.ActionUi.reload_detection()
                self.start.ActionUi.save_profile()
            else:
                messagebox.showwarning("Please select roi on left", "Warning")
            self.start.MainUi.roi_setting_frame.refresh_roi_settings()
        
    def refresh(self):
        if self.start.MainUi.roi_index > -1 :
            
            roi = self.start.Profile.loaded_profile[self.start.MainUi.cam_pos][self.start.MainUi.cmb_pos]["roi"][self.start.MainUi.roi_index]
            
            #Add On for sub type
            if roi.get("mm_pixel") is None:
                roi.update({"mm_pixel":1.0})
                print(f"adding mm_pixel {str(roi.get('mm_pixel'))}")
                self.start.Main.flag_reload = True
                self.start.Profile.flag_save = True
                
            #Add On for sub type
            if roi.get("blur") is None:
                roi.update({"blur":30})
                print(f"adding blur {str(roi.get('blur'))}")
                self.start.Main.flag_reload = True
                self.start.Profile.flag_save = True
                
            if roi.get("method") is None:
                roi.update({"method":5})
                print(f"adding method {str(roi.get('method'))}")
                self.start.Main.flag_reload = True
                self.start.Profile.flag_save = True
                
            if roi.get("mask") is None:
                img = roi.get('img')
                image_height = img.shape[0]
                image_width = img.shape[1]
                number_of_color_channels = 1
                color = (255)
                pixel_array = numpy.full((image_height, image_width, number_of_color_channels), color, dtype=numpy.uint8)
                
                roi.update({"mask_64":base64.b64encode(cv2.imencode('.png',copy.deepcopy(pixel_array))[1]).decode()})
                roi.update({'mask':copy.deepcopy(pixel_array)})
                self.start.Main.flag_reload = True
                self.start.Profile.flag_save = True
                
            if roi.get("angle_enable") is None:
                roi.update({"angle_enable":True})
                self.start.Main.flag_reload = True
                self.start.Profile.flag_save = True
            
            if roi.get("auto_move") is None:
                roi.update({"auto_move":False})
                self.start.Main.flag_reload = True
                self.start.Profile.flag_save = True
             
            self.enable_auto_move_var.set(roi.get("auto_move"))
                
            self.enable_angle_var.set(roi.get("angle_enable"))
              
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
