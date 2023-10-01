import webbrowser
import numpy as np
import cv2
import time

def init(s):
    global self
    self = s 
    

def always_show_on_bottom():
    self.config.MODE.always_show_on_bottom = not self.config.MODE.always_show_on_bottom

def on_off_live(off_live = False, on_off = None):
    if self.ViewEvent.live or on_off == True:
        self.ViewEvent.live = False
        self.ViewEvent.off_live = off_live
    elif not self.ViewEvent.live or on_off == False:
        self.ViewEvent.live = True
        self.ViewEvent.off_live = False
        self.MainUi.view.scale = 1
        self.MainUi.view.after(33, self.ViewEvent.update_view)
    self.MainUi.refresh()

def change_profile(text):
    if self.Users.login_admin():
        self.Profile.select_profile(text)
        self.Config.config.PROFILE.name = self.Profile.name
        self.Config.profile = self.Profile.name
        self.CNN.init()
        self.ANO.init()
        reset_detection_step()
                

def cnn_train():
    if self.Users.login_admin():
        self.CNN.train()

def ano_train():
    if self.Users.login_admin():
        self.ANO.train()
   
def save_profile():
    self.Profile.write()
    
def go_next():
    print("Robot Next Step")
    self.Com.go_next()
    print("Skip Step")
    #self.Main.next_step()
    self.Main.detect(self.Main.detected_step+1)
    self.MainUi_Refresh.refresh_main_ui()

def skip_step():
    print("Skip Step")
    self.Main.next_step()
    self.MainUi_Refresh.refresh_main_ui()

def detect(SN=""):
    print("detect with serial no")
    print("cmb.current() {} detected_step{}".format(self.MainUi.cmb.current(),self.Main.detected_step))
    self.Main.detect(SN=SN)
    self.MainUi_Refresh.refresh_main_ui()
    self.MainUi.write('Current Detection Step is {}'.format(self.Main.detected_step))

def detect_with_frames(in_SN="", in_frames=None):
    print("detect with serial no and frames")
    print("cmb.current() {} detected_step{}".format(self.MainUi.cmb.current(),self.Main.detected_step))
    self.Main.detect(SN=in_SN, in_frames = in_frames)
    self.MainUi_Refresh.refresh_main_ui()
    self.MainUi.write('Current Detection Step is {}'.format(self.Main.detected_step))

def detect_step(step : int, SN=""):
    print("detect by step")
    print("cmb.current() {} detected_step{}".format(self.MainUi.cmb.current(),self.Main.detected_step))
    self.Main.detect(step,SN=SN)
    self.MainUi_Refresh.refresh_main_ui()
    self.MainUi.write('Current Detection Step is {}'.format(self.Main.detected_step)) 

def detect_current_step():
    print("detect current step")
    print("cmb.current() {} detected_step{}".format(self.MainUi.cmb.current(),self.Main.detected_step))
    self.Main.detect(step = self.MainUi.cmb.current())
    self.MainUi_Refresh.refresh_main_ui()
    self.MainUi.write('Current Detection Step is {}'.format(self.Main.detected_step))

def reset_detection_step():
    self.Main.reset()
    self.Com.stop()
    self.MainUi_Refresh.refresh_main_ui()

def reload_detection():
    self.Main.reload()
    self.Main.current_step = self.MainUi.cmb.current()
    self.MainUi_Refresh.refresh_main_ui()

def open_folder(folder_name):
    if self.Users.login_user():
        pth = self.Profile.get_image_folder_path(folder_name)
        webbrowser.open(pth)
        
def open_profile_folder():
    if self.Users.login_admin():
        pth = self.Profile.get_profile_folder_path()
        webbrowser.open(pth)

def open_log_folder():
    webbrowser.open(str(self.Config.results_path))

import subprocess
from pathlib import Path

def open_image_log_folder():
    if  Path("C:\\Program Files\\Everything\\Everything.exe").is_file():
        p = subprocess.run(["C:\\Program Files\\Everything\\Everything.exe",
                    "-path",str(self.Config.images_path),
                    "-sort","Date Created",
                    "-sort-descending","",
                    "-thumbnail-size", "300",
                    "-thumbnails","",
                    "-filter", "Picture",
                    "-search", self.Main.barcode])
    else:
        webbrowser.open(self.Config.images_path )

