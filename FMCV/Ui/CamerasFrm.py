from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image

import cv2

from FMCV.Ui.ScrollableFrm  import ScrollableFrame

from FMCV import Util
import os

import traceback

class CamerasFrame(ttk.Frame):

    dt_frm_cameras = {}

    def __init__(self, start, parent, *args, **kwargs):
        #https://stackoverflow.com/questions/17466561/best-way-to-structure-a-tkinter-application/17470842
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.start = start
        
        sf = ScrollableFrame(self)        
        sf.pack(fill = BOTH, expand = True)
        self.frm = frm = sf.scrollable_frame
        
        if start.Config.BRAND == "NA":
            self.img = None
            if self.img is None:
                self.img = ImageTk.PhotoImage(Image.open(os.path.join("FMCV","Ui","fmcv-square-128.png")))
                canvas_cam = Canvas(self.frm, highlightthickness = 1, 
                                                    highlightbackground = "white",
                                                    relief = GROOVE,
                                                    borderwidth = 2,
                                                    width=128,
                                                    height=128
                                                    )
                canvas_cam.pack()
                #canvas_cam.image = self.img
                canvas_cam.create_image(3, 3, image=self.img, anchor='nw')
            
            
    def cv2_to_pil(self,cv2_image):
        #https://stackoverflow.com/questions/43232813/convert-opencv-image-format-to-pil-image-format/48602446
        img = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        return im_pil
        
    def update(self,images):
        try:
            size = 128
            for name,image in images.items():
                canvas_cam = self.dt_frm_cameras.get(name)
                if canvas_cam is None:
                    canvas_cam = Canvas(self.frm, highlightthickness = 1, 
                                                highlightbackground = "white",
                                                relief = GROOVE,
                                                borderwidth = 2,
                                                width = size,
                                                height = size)
                    canvas_cam.pack()
                    canvas_cam.bind("<Button-1>",lambda event,name = name:self.camera_clicked(event,name))
            
                    Label(self.frm, text = f'{name}').pack()
                    self.dt_frm_cameras.update({name:canvas_cam})
                
                if image is not None:
                    image = cv2.resize(image,(size,size))
                    image=self.cv2_to_pil(image)
                    canvas_cam.image = None
                    canvas_cam.delete("all")
                    canvas_cam.image = ImageTk.PhotoImage(image)
                    canvas_cam.create_image(3, 3, image=canvas_cam.image, anchor='nw')
            
 
        except:
            traceback.print_exc()
        
    def camera_clicked(self,event, in_name):
        #print("{} {}".format(event,in_name))
        S = self.start
        
        for n,(name,view) in enumerate(self.dt_frm_cameras.items()):
            if in_name == name: 
                S.MainUi.cam_pos = n
                S.MainUi.cmb_cam.current(n)
                S.MainUi_Refresh.refresh_main_ui()
                #print(n)
        
        
    def update_results(self):
        S = self.start

        for n,(name,view) in enumerate(self.dt_frm_cameras.items()):
            view.config(highlightbackground='white')
            
            try : 
                is_pass = True
                color = 'green'
                if n < (len(S.Main.results)):
                    if S.MainUi.cmb_pos < len(S.Main.results[n]):
                        for roi_n, roi in enumerate(S.Main.results[n][S.MainUi.cmb_pos]):
                            roi_pass = roi.get('PASS')
                            #print(f"{n} {S.MainUi.cmb_pos} {roi_pass}")
                            if roi_pass is True:
                                color = 'green'
                            else:
                                color = 'red'
                                break
                        view.config(highlightbackground=color)
            except:
                traceback.print_exc()
                print("Empty results")