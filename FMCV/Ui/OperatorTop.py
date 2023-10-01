from tkinter import *
from tkinter import ttk

from FMCV.Ui.CamerasFrm import CamerasFrame

class OperatorWindow(Toplevel):
     
    def __init__(self,start, master = None):
         
        super().__init__(master = master)
        self.title("New Window")
        self.geometry("200x200")
        label = Label(self, text ="This is a new Window")
        label.pack()
        frm_cams = CamerasFrame(start, self)
        frm_cams.pack()