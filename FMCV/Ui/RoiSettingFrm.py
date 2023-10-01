from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import ImageTk, Image
import cv2
import base64
import copy
from FMCV.Cv import Cv
from FMCV.Ui.ScrollableFrm  import ScrollableFrame
from FMCV import Util
from FMCV.Ui.ImageViewFrm import ImageView
from FMCV.Ui.RoiSettingCnnFrm import ROISettingCNNFrame
from FMCV.Ui.RoiSettingAnoFrm import ROISettingANOFrame
from FMCV.Ui.RoiSettingFiducialFrm import ROISettingFiducialFrame
from FMCV.Ui.RoiSettingBarcodeFrm import ROISettingBarcodeFrame
from FMCV.Ui.RoiSettingOcrFrm import ROISettingOcrFrame
from FMCV.Ui.RoiSetting3dFrm import ROISetting3dCalibrationFrame
from FMCV.Ui.RoiSettingCobotFrm import ROISettingCobotFrame
from FMCV.Ui.RoiSettingLocateFrm import ROISettingLocateFrame
from FMCV.Ui.RoiSettingDistantFrm import ROISettingDistantFrame

class ROISettingFrame(ttk.Frame):

    def __init__(self,start, parent, *args, **kwargs):
        #https://stackoverflow.com/questions/17466561/best-way-to-structure-a-tkinter-application/17470842
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.start = start
        
        m1 = ttk.PanedWindow(self,orient=HORIZONTAL)
        m1.pack(fill=BOTH, expand=True)
        
        self.setting_frame = ScrollableFrame(m1)
        self.setting = setting = self.setting_frame.scrollable_frame
        
        self.lbl_roi_name = Label(setting, text = 'N/A')
        self.lbl_roi_name.pack()
        
        Label(setting, text = 'Result as source').pack()
        self.result_source_entry = Entry(setting)
        self.result_source_entry.pack()
        self.result_source_entry.bind('<KeyRelease>',self.save_result_as_source)
        
        Label(setting, text = 'Rotate').pack()
        self.rotate_cmb = rotate_cmb = ttk.Combobox(setting,state="readonly")
        rotate_cmb['values'] = ("","90","180","270","F","F90","F180","F270")
        rotate_cmb.pack()
        rotate_cmb.current(0)
        rotate_cmb.bind("<<ComboboxSelected>>", self.roi_rotate_cmb_callback)
        
        self.btn_r_image_update = Button(setting,text ="Roi Image Update")
        self.btn_r_image_update.pack(side=TOP)
        self.btn_r_image_update.configure(command=self.update_roi_image) 
        
        btn_frm = ttk.Frame(setting)
        btn_frm.pack()
        btn_r_image_download = Button(btn_frm,text ="Download Roi", command = self.download_roi_image)
        btn_r_image_download.pack(side=LEFT)
        
        btn_r_image_download = Button(btn_frm,text ="Upload Roi", command = self.upload_roi_image)
        btn_r_image_download.pack(side=RIGHT)
        
        Label(setting, text = 'ROI TYPE').pack()
        self.roi_cmb = roi_cmb = ttk.Combobox(setting,state="readonly")
        roi_cmb['values'] = ("AI","AI2","QR","OCR","2D","LOC","DIST")#("CNN","FIDUCIAL","ANO","QR","OCR")
        roi_cmb.pack()
        roi_cmb.current(0)
        roi_cmb.bind("<<ComboboxSelected>>", self.roi_type_cmb_callback)
        
        #CNN Frame
        self.cnn_frame = ROISettingCNNFrame(self.start,setting)
        
        #ANO Frame
        self.ano_frame = ROISettingANOFrame(self.start,setting)
        
        #Fiducial Frame
        self.fiducial_frame = ROISettingFiducialFrame(self.start,setting)
        
        #3D Calibration Frame
        self.calibration_3d_frame =  ROISetting3dCalibrationFrame(self.start,setting)
        
        #Barcode Frame
        self.barcode_frame = ROISettingBarcodeFrame(self.start,setting)
        
        #OCR Frame
        self.ocr_frame = ROISettingOcrFrame(self.start,setting)
        
        #MOVE Frame
        self.move_frame = ROISettingCobotFrame(self.start,setting)
        
        #LOCATE Frame
        self.locate_frame = ROISettingLocateFrame(self.start,setting)
        
        #LOCATE Frame
        self.distant_frame = ROISettingDistantFrame(self.start,setting)
        
        self.image_view = ImageView(self, relief = GROOVE, borderwidth = 2)
        m1.add(self.image_view, weight = 10)

        m1.add(self.setting_frame, weight = 5)
        
        self.id_picture = -1
        self.image = None
        self.scale = 1
    
        #Disable button when is not ENGINEERING mode
        if start.Config.mode_name != "ENGINEER":
            self.btn_r_image_update["state"] = DISABLED
            btn_r_image_download.config(state = "disable")
            
    def save_result_as_source(self,event):
        #print(event)
        if not self.start.Users.login_admin():
            return
            
        ret, roi = self.start.Profile.get_selected_roi()
        if not ret:
            self.image_view.clear()
            return 
            
        if self.start.Config.mode_name != "ENGINEER":
            print("please enable engineer mode")
            return 
        #print(event.widget.get())
        roi.update({"source":event.widget.get()})
        self.refresh_roi_settings()
        
    def roi_type_cmb_callback(self,event):
        if self.start.Users.login_admin():
            if self.start.Config.mode_name == "ENGINEER":
                if self.start.MainUi.roi_index > -1:
                    roi = self.start.Profile.loaded_profile[self.start.MainUi.cam_pos][self.start.MainUi.cmb_pos]["roi"][self.start.MainUi.roi_index]
                    roi.update({"type":event.widget.get()})
                    
                    self.start.Main.flag_reload = True
                    self.start.Profile.flag_save = True
            else:
                print("please enable engineer mode")
            self.display_roi_type_widget()
            
    def roi_rotate_cmb_callback(self,event):
        if not self.start.Users.login_admin():
            return
            
        ret, roi = self.start.Profile.get_selected_roi()
        if not ret:
            self.image_view.clear()
            return 
            
        if self.start.Config.mode_name != "ENGINEER":
            print("please enable engineer mode")
            return 
        
        roi.update({"rotate":event.widget.get()})
        
        if not self.start.ViewEvent.live or self.start.ViewEvent.source:
            self.start.MainUi.view.set_rotate(event.widget.get())
            self.start.MainUi.view.refresh_image()

        self.start.Main.flag_reload = True
        self.start.Profile.flag_save = True


    def clear_roi_type_widget(self):
        self.cnn_frame.pack_forget()
        self.ano_frame.pack_forget()
        self.fiducial_frame.pack_forget()
        self.barcode_frame.pack_forget()
        self.ocr_frame.pack_forget()
        self.calibration_3d_frame.pack_forget()
        self.move_frame.pack_forget()
        self.locate_frame.pack_forget()
        self.distant_frame.pack_forget()
        
    def display_roi_type_widget(self):        
        #Clear frames
        self.clear_roi_type_widget()
        
        if self.start.MainUi.roi_index > -1:
            roi_type = self.start.Profile.loaded_profile[self.start.MainUi.cam_pos][self.start.MainUi.cmb_pos]["roi"][self.start.MainUi.roi_index].get('type')
            if roi_type in ("AI","CNN"):
                self.cnn_frame.pack(fill=BOTH, expand=True)
                self.cnn_frame.refresh()
            if roi_type in ("AI2","ANO"):
                self.ano_frame.pack(fill=BOTH, expand=True)
                self.ano_frame.refresh()
            if roi_type in ("2D","FIDUCIAL"):
                self.fiducial_frame.pack(fill=BOTH, expand=True)
                self.fiducial_frame.refresh()
            if roi_type in ("3D"):
                self.calibration_3d_frame.pack(fill=BOTH, expand=True)
                self.calibration_3d_frame.refresh()
            if roi_type == "QR":
                self.barcode_frame.pack(fill=BOTH, expand=True)
                self.barcode_frame.refresh()
            if roi_type == "OCR":
                self.ocr_frame.pack(fill=BOTH, expand=True)
                self.ocr_frame.refresh()    
            if roi_type == "MOVE":
                self.move_frame.pack(fill=BOTH, expand=True)
                self.move_frame.refresh()
            if roi_type == "LOC":
                self.locate_frame.pack(fill=BOTH, expand=True)
                self.locate_frame.refresh()
            if roi_type == "DIST":
                self.distant_frame.pack(fill=BOTH, expand=True)
                self.distant_frame.refresh()
                
                
    def refresh_roi_settings(self):
        ret, roi = self.start.Profile.get_selected_roi()
        if not ret:
            self.lbl_roi_name.config(text = "N/A")
            self.image_view.clear()
            return 
        
        # Update name
        self.lbl_roi_name.config(text = f"ROI Name : {roi.get('name')}")
        
        # Update rotation setting
        for n, text in enumerate(self.rotate_cmb['values']):
            if text == roi.get("rotate"):
                self.rotate_cmb.current(n)
        
        # Update type  
        roi_type = roi.get("type")
        if roi_type in ("AI","CNN"):
            self.roi_cmb.set("AI")
        elif roi_type in ("2D","FIDUCIAL"):
            self.roi_cmb.set("2D")
        elif roi_type in ("AI2","ANO"):
            self.roi_cmb.set("AI2")
        else:
            self.roi_cmb.set(roi_type)

        #for n, text in enumerate(self.roi_cmb['values']):
        #    if text == roi.get("type"):
        #        self.roi_cmb.current(n)
                
        # Update setting        
        self.display_roi_type_widget()
        
        # Update source
        self.result_source_entry.delete(0, END)
        if roi.get("source") is None:
            if self.start.ViewEvent.source:
                self.start.MainUi.view.scale = 1 
            self.start.ViewEvent.update_source(False)
            return
            
        self.result_source_entry.insert(0, str(roi.get("source")))
        
        if roi.get("source") != "":
            if not self.start.ViewEvent.source:
                self.start.ViewEvent.update_source(True)
        else:
            if self.start.ViewEvent.source:
                self.start.MainUi.view.scale = 1 
            self.start.ViewEvent.update_source(False)
        

    def update_roi_image(self):
        if not self.start.Users.login_admin():
            return
            
        ret, roi = self.start.Profile.get_selected_roi()
        if not ret:
            self.image_view.clear()
            return 
        
        if self.start.ViewEvent.live and not self.start.ViewEvent.source:
            ret, frame = self.start.Profile.get_selected_roi_frame()
        else:
            ret, frame = self.start.MainUi.view.get_image()
           
        if not ret:
            return
           
        x1 = roi['x1']
        y1 = roi['y1']
        x2 = roi['x2']
        y2 = roi['y2']
        
        roi.update({"image":base64.b64encode(cv2.imencode('.png',copy.deepcopy(frame[y1:y2,x1:x2]))[1]).decode()})
        roi.update({'img':copy.deepcopy(frame[y1:y2,x1:x2])})
        self.image_view.set_image(roi['img'])
        self.start.Main.flag_reload = True
        self.start.Profile.flag_save = True

            
    def download_roi_image(self):
        roi_index = self.start.MainUi.roi_index
        if roi_index > -1:
            roi = self.start.Profile.loaded_profile[self.start.MainUi.cam_pos][self.start.MainUi.cmb_pos]["roi"][roi_index]
            f = filedialog.asksaveasfilename(initialfile = 'roi.png',defaultextension=".bmp",filetypes=[("All Files","*.*"),("bitmap","*.bmp"),("PNG","*.png")])
            print(f)
            cv2.imwrite(f, roi['img'])
            
    def upload_roi_image(self):
        if self.start.Users.login_admin():
            roi_index = self.start.MainUi.roi_index
            if roi_index > -1:
                roi = self.start.Profile.loaded_profile[self.start.MainUi.cam_pos][self.start.MainUi.cmb_pos]["roi"][roi_index]
                file_path = filedialog.askopenfilename(filetypes=(("Image files", "*.jpg;*.bmp;*.png"),("All files", "*.*") ))
                img = cv2.imread(file_path)
                if img is not None:
                    roi.update({"image":base64.b64encode(cv2.imencode('.png',copy.deepcopy(img))[1]).decode()})
                    roi.update({'img':copy.deepcopy(img)})
                    self.image_view.set_image(roi['img'])
                    self.start.Main.flag_reload = True
                    self.start.Profile.flag_save = True