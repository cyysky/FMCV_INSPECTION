from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image
import cv2
from FMCV.Ui.ScrollableFrm  import ScrollableFrame
from tkinter import simpledialog, messagebox
from FMCV.Cv import Match,Cv
from tkinter import simpledialog
import time
import random
import numpy as np
from FMCV import Logging

#import tracemalloc
  
#tracemalloc.start()
  
class MainMenu(Menu):
    def __init__(self, start, parent, *args, **kwargs):
        global calibrated
        calibrated = None
    
        #https://stackoverflow.com/questions/17466561/best-way-to-structure-a-tkinter-application/17470842
        Menu.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.start = start
        
        menubar = self
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open Profile Folder", command=self.start.ActionUi.open_profile_folder)
        filemenu.add_command(label="New Profile", command=self.add_profile)
        filemenu.add_command(label="Load AI & Profile Configuration", command=self.load)
        filemenu.add_command(label="Save Configuration", command=self.start.Config.save)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.start.MainUi.on_closing)#parent.quit)
        menubar.add_cascade(label="File", menu=filemenu)
        
        viewmenu = Menu(menubar, tearoff=0)
        
        menubar.add_cascade(label="Edit", menu=viewmenu)
        viewmenu.add_command(label="ROI Busket", command=lambda:self.pub("ui/roi_busket/show"))
        viewmenu.add_command(label="Search similar ROI", command=lambda:self.pub("ui/roi_search/show"))
        viewmenu.add_command(label="Reset ROI Position", command=self.reset_roi_position)
        viewmenu.add_command(label="Set ROI to Center", command=self.start.MainUi.move_roi_center)
        viewmenu.add_command(label="Make ROI square for AI", command=self.start.MainUi.make_roi_square)
        viewmenu.add_command(label="Save AI Model",command=self.start.CNN.save_model)
        viewmenu.add_command(label="Train AI Model",command=self.start.CNN.train)

        if start.Config.log_type in ("PLEXUS","KAIFA"):
            viewmenu.add_command(label="Users Managements", command=self.users_management)

        viewmenu = Menu(menubar, tearoff=0)
        viewmenu.add_command(label="Pause/Resume Live Frame", command=lambda:self.start.ActionUi.on_off_live(False))
        viewmenu.add_command(label="View Results", command=self.view_results)
        viewmenu.add_command(label="Toggle always show on bottom", command=self.start.ActionUi.always_show_on_bottom)
        viewmenu.add_command(label="Toggle Live View", command=lambda:self.start.ActionUi.on_off_live(True))
        viewmenu.add_command(label="show step all ROIs", command=self.start.MainUi.display_other_roi)
        viewmenu.add_command(label="show ROI margin", command=self.start.MainUi.display_roi_margin)
        menubar.add_cascade(label="View", menu=viewmenu)

        reportmenu = Menu(menubar, tearoff=0)
        reportmenu.add_command(label="Download Failed Summary", command=self.start.Log.write_excel)
        menubar.add_cascade(label="Report", menu=reportmenu)
        
        #RnD = Menu(menubar, tearoff=0)
        #RnD.add_command(label="Development RUN Program", command=self.development_send_run)
        #RnD.add_command(label="Development Update ROI ", command=lambda:self.pub("tool/RoiPositionScaleUpdate/open_dialog"))
        #RnD.add_command(label="print mem alloc", command=self.print_mem_alloc)
        #menubar.add_cascade(label="R&D", menu=RnD)
        
        tools = Menu(menubar, tearoff=0)
        if start.Config.debug_level < 10 :
            # Disabled as not stable
            tools.add_command(label="Robot Tool", command=self.robot_tools)
       
        tools.add_command(label="Verify AI", command=lambda:self.pub("cnn/verify"))
        tools.add_command(label="Verify AI(Single)", command=lambda:self.pub("cnn/verify_single"))
        menubar.add_cascade(label="Tools", menu=tools)

        helpmenu = Menu(menubar, tearoff=0)
        helpmenu.add_command(label="Help Index", command=self.donothing)
        helpmenu.add_command(label="About...", command=self.about)
        menubar.add_cascade(label="Help", menu=helpmenu)
        
    def print_mem_alloc(self):
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
          
        for stat in top_stats[:100]:
           print(stat)

        
    def donothing(self,*args, **kwargs):
        print(args)
        print(kwargs)
        
    def pub(self,text):
        self.start.pub(text)
    
    def load(self):
        if self.start.Users.login_admin():
            self.start.Config.read()
            self.start.CNN.init()
            self.start.ANO.init()
            self.start.Profile.init()
            self.start.ActionUi.reset_detection_step()
        
    def add_profile(self):
        if self.start.Profile.flag_save:
            if messagebox.askokcancel("Save Profile Setting", f"Do you want to save profile {self.start.Profile.name} settings?"):
                self.start.Profile.write()
        
        if not self.start.Profile.flag_save:
            text = simpledialog.askstring("New Profile", "Enter New Profile Name:")
            if text:
                self.start.ActionUi.change_profile(text)
                
                Logging.info(f"added profile {text}")
            
    def about(self):
        self.start.pub("ui/help/about",info="hello world")
        
    def view_results(self):
        self.start.pub("ui/results/show")
        
    def users_management(self):
        self.start.pub("ui/users/show")
        
    def reset_roi_position(self):
        self.start.MainUi_Edit.reset_roi_position()
        
    def development_send_run(self):
        #self.start.Com.send_tcp_command(self.start.Profile.name)
        self.start.host.broadcast_message("hello")
        pass

    def power_on_robot(self):
        if self.start.Users.login_admin():
            self.start.Com.power_on()
            
    def robot_tools(self):
        if self.start.Users.login_admin():
            if self.start.config.CONTROL.tcp_type.casefold() == "jaka":
                self.start.pub("platform/jaka/show")
            else:
                messagebox.showwarning(title="Robot tools", message="No setup")