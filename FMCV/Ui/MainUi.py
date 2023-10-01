import os
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from tkinter.messagebox import askokcancel, showinfo, WARNING
from FMCV.Ui.ImageViewFrm import ImageView
from FMCV.Ui.ResultsFrm import ResultsFrame
from FMCV.Ui.ScrollableFrm  import ScrollableFrame
from FMCV.Ui.RoiSettingFrm import ROISettingFrame
from FMCV.Ui.CamerasFrm import CamerasFrame
from FMCV.Ui.MainUi_Control import ControlFrame
from FMCV.Ui.MenuFrm import MainMenu
from FMCV import Logging
from FMCV.Cv import Cv

from tkinter import messagebox

import traceback
import ctypes

def is_packed(widget):
    try:
        widget.pack_info()
        return True
    except TclError:
        return False

top = Tk()

roi_index = -1
cmb_pos = 0
cam_pos = 0

id_rect = -1
# Code to add widgets will go here...
lx = -1
ly = -1

mv_type = ""
window_width, window_height = 0, 0

def on_focus(event):
    if start.config.MODE.always_show_on_bottom:
        top.lower()

def on_closing():
    global self
    global top
    # lambda e: top.quit()
    if self.Profile.flag_save:
        if messagebox.askokcancel("Save", "Do you want to save settings?"):
            self.Profile.write()
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        top.destroy()

def init(s):
    global self
    global start
    self = s 
    start = s 
    
    global top, view, r_view, roi_setting_frame #.t_view
    
    global cf, Lb1, cmb_cam, cmb, roi_entry, barcode_entry, cmb_profile
    #cf is controlframe
    
    global btn_add_roi, btn_remove_roi, btn_rename_roi
    
    global btn_add_step, btn_remove_step, btn_add_step_above
    
    global btn_add_source, btn_remove_source
    
    global result_frame
    
    global statusbar
    
    global frm_cams
    
    global steps_lbl
    
    # https://stackoverflow.com/questions/15981000/tkinter-python-maximize-window
    w = top.winfo_screenwidth()
    if os.name == 'nt':
        top.state('zoomed')
        h = top.winfo_screenheight() - 30
    else:
        h = top.winfo_screenheight() - 65
    top.geometry("{}x{}-0+0".format(w, h))   
    top.lift()
    top.attributes('-topmost', 1)
    top.attributes('-topmost', 0)
    
    title = "FMCV AI Deep Learning Vision Inspection"

    top.title(f"{title} version {self.Config.version}, Selected Profile : {self.Config.profile}")
    
    statusbar = Label(top, text="Version {}".format(start.Config.version), bd=1, relief=SUNKEN, anchor=W)
    statusbar.pack(side=BOTTOM, fill=X)
    
    m1 = ttk.PanedWindow(top, orient=HORIZONTAL)
    m1.pack(fill=BOTH, expand=True)
    
    paned_left = ttk.PanedWindow(m1, orient=HORIZONTAL)
    
    cf = ControlFrame(self,paned_left)
    Lb1 = cf.Lb1
    Lb1.bind("<<ListboxSelect>>", Lbl_callback)
    cmb_profile = cf.cmb_profile
    cmb_profile.bind("<<ComboboxSelected>>", cmb_profile_callback)
    
    cmb_cam = cf.cmb_cam
    cmb_cam.bind("<<ComboboxSelected>>", cmb_callback)
    cmb = cf.cmb
    cmb.bind("<<ComboboxSelected>>", cmb_callback)
    roi_entry = cf.roi_entry
    btn_add_roi = cf.btn_add_roi
    btn_remove_roi = cf.btn_remove_roi
    btn_rename_roi = cf.btn_rename_roi
    btn_add_step = cf.btn_add_step
    btn_add_step_above = cf.btn_add_step_above
    btn_remove_step = cf.btn_remove_step
    btn_add_source = cf.btn_add_source
    btn_remove_source = cf.btn_remove_source
    steps_lbl = cf.steps_lbl
    barcode_entry = cf.barcode_entry

    m2 = ttk.PanedWindow(m1, orient=VERTICAL)
    
    m3 = ttk.PanedWindow(m2, orient=HORIZONTAL)
    m4 = ttk.PanedWindow(m2, orient=HORIZONTAL)
    
    view  = ImageView(m2,refresh, relief = GROOVE, borderwidth = 2)
    r_view = ImageView(m2,refresh, relief = GROOVE, borderwidth = 2)

    m3.add(view, weight=1)
    m3.add(r_view, weight=1)
    
    result_frame = ResultsFrame(start,m4)
    
    frm_cams = CamerasFrame(start, paned_left)
    roi_setting_frame = ROISettingFrame(self,m4)
    m4.add(roi_setting_frame, weight=10)
    m4.add(result_frame, weight=12)

    bottom_frame = ttk.PanedWindow(m2, orient=HORIZONTAL)
    platform_control_frame = LabelFrame(bottom_frame, text="Platform Control")
    bottom_frame.add(platform_control_frame, weight=1)
    platform_control_frame.columnconfigure(0, weight=1)
    platform_control_frame.columnconfigure(1, weight=1)
    platform_control_frame.columnconfigure(2, weight=1)
    platform_control_frame.columnconfigure(3, weight=1)

    self.platform_mode = StringVar()
    self.platform_mode_label = Label(platform_control_frame, textvariable=self.platform_mode)
    self.platform_mode_label.grid(row=0, column=0, padx=5, pady=5, sticky='ew')
    self.platform_enable_button = Button(platform_control_frame, text="Enable Platform", command=toggle_platform_mode)
    self.platform_enable_button.grid(row=0, column=1, padx=5, pady=5, sticky='ew')
    self.platform_clear_error_button = Button(platform_control_frame, text="Clear Error", command=clear_platform_error)
    self.platform_clear_error_button.grid(row=0, column=2, padx=5, pady=5, sticky='ew')
    self.platform_reset_button = Button(platform_control_frame, text="Reset", command=reset_platform)
    self.platform_reset_button.grid(row=0, column=3, padx=5, pady=5, sticky='ew')

    m2.add(m3, weight=10)
    m2.add(m4, weight=10)
    if start.Config.platform_model != "NONE":
        m2.add(bottom_frame, weight=1)
    
    paned_left.add(cf, weight = 10)
    paned_left.add(frm_cams, weight = 7)
    
    weight = 27

    
    m1.add(paned_left,weight=weight)
    m1.add(m2,weight=100)
    
    top.protocol("WM_DELETE_WINDOW", on_closing)
    top.bind('<Escape>',lambda v: on_closing())
    top.bind("<Configure>", configure_event)
    top.bind("<Control-Shift-space>", key_control_space_action)
    top.bind("<Control-space>", key_space_action)
    top.bind('<Return>',key_return_action)
    top.bind("<Control-D>", display_other_roi)
    top.bind("<Control-M>", display_roi_margin)
    top.bind("<Control-O>", make_roi_square)
    top.bind("<Control-C>", move_roi_center)
    top.bind('<FocusIn>', on_focus)
    top.config(menu=MainMenu(start,top))
    
    start.sub("ui/roi/display_other_roi", display_other_roi)
    start.sub("ui/roi/display_roi_margin", display_roi_margin)

    start.sub("ui/roi/make_roi_square", make_roi_square)
    start.sub("ui/roi/move_roi_center", move_roi_center)

def refresh():
    start.MainUi_Refresh.refresh_main_ui()
    start.MainUi.top.update()
    
def update_result_frame(img):
    start.MainUi.r_view.set_image(img)
    start.MainUi.top.update()

def display_other_roi(*args): #Prevent partially initialized module error
    start.MainUi_Refresh.display_other_roi()
def display_roi_margin(*args): #Prevent partially initialized module error
    start.MainUi_Refresh.display_roi_margin()
def make_roi_square(*args): #Prevent partially initialized module error
    start.MainUi_Edit.make_roi_square()
def move_roi_center(*args): #Just follow above 
    start.MainUi_Edit.move_roi_center()

def key_control_space_action(*args):
    start.ActionUi.on_off_live()
    
def key_return_action(*args):
    barcode_entry.select_range(0, 'end')
    
def key_space_action(*args):
    Logging.info(args)
    Logging.info("Is cnn widget widget mapped ",is_packed(roi_setting_frame.cnn_frame))
    if start.config.MODE.spacebar_save_roi and is_packed(roi_setting_frame.cnn_frame):
        roi_setting_frame.cnn_frame.save_roi_image()

def write(text):
    global statusbar
    statusbar.config(text=text)

def cmb_callback(event):
    global cmb_pos, cam_pos
    cam_pos = cmb_cam.current()
    cmb_pos = cmb.current()
    Logging.debug(f"step combobox callback {event.widget.current()} {event.widget.get()}")

    self.Main.current_step = cmb_pos
    self.Main.detected_step = cmb_pos
    
    self.Main.move_platform_to_position(cmb_pos)

    self.MainUi_Refresh.refresh_main_ui()
 
def cmb_profile_callback(event):
    if self.Profile.flag_save:
        if messagebox.askokcancel("Save Profile Setting", f"Do you want to save profile {self.Profile.name} settings?"):
            self.Profile.write()
    
    if not self.Profile.flag_save:
        self.ActionUi.change_profile(cmb_profile.get()) 
        Logging.info(f"cmb_profile_callback {cmb_profile.get()}")
        
    start.ActionUi.reset_detection_step()
    
def event_unbind():
    Lb1.unbind("<<ListboxSelect>>")

def event_bind():
    Lb1.bind("<<ListboxSelect>>", Lbl_callback)

def Lbl_callback(event):
    global roi_index
    selection = event.widget.curselection()
    Logging.debug(f"ROI callback {selection}")
    if selection:
        roi_index = selection[0]
        data = event.widget.get(roi_index)
        self.MainUi_Refresh.refresh_main_ui()

def reset_roi_index():
    global roi_index
    roi_index = -1
    self.MainUi_Refresh.refresh_edit_roi_rectangle()
    self.MainUi_Refresh.display_roi_image()
    roi_setting_frame.refresh_roi_settings()

def refresh_roi_index():
    self.MainUi_Refresh.refresh_edit_roi_rectangle()
    self.MainUi_Refresh.display_roi_image()
    roi_setting_frame.refresh_roi_settings()

    
def configure_event(event):
    #https://stackoverflow.com/questions/61712329/tkinter-track-window-resize-specifically
    global window_width, window_height
    try:
        if event.widget == top:
            if (window_width != event.width) and (window_height != event.height):
                window_width, window_height = event.width,event.height
                print(f"The width of Toplevel is {window_width} and the height of Toplevel "
                      f"is {window_height}")
                      
        self.MainUi_Refresh.refresh_edit_roi_rectangle()
        
        if top.state() == 'zoomed':
            #print("My window is maximized")
            pass
        if top.state() == 'normal':
            #print("My window is normal")
            pass
    except:
        pass
        
last_scale = 1
def update_source(frames):
    global cam_pos, view
    global last_scale
    try:
        frm = list(frames.values())[cam_pos]
        if roi_index > -1:
            degree = start.Profile.loaded_profile[cam_pos][cmb_pos]['roi'][roi_index]['rotate']
            #frm = Cv.get_rotate(degree,frm)
            view.set_rotate(degree)
        else:
            view.set_rotate("")
        view.set_image(frm)
        
        if last_scale != view.get_scale():
            start.MainUi_Refresh.refresh_edit_roi_rectangle()
        last_scale = view.get_scale()
    except:
        print("update source exception")
        #traceback.print_exc()
        
def ask_reset_continuous_fail_alarm():
    answer = askokcancel(title='Alert',
        message='Fail 3 times, still continue?',
        icon=WARNING)
    return answer
        # showinfo(
            # title='Deletion Status',
            # message='The data is deleted successfully')

def update_platform_status(is_connected=False, mode=""):
    """Get the platform status from Main.py and update it to UI"""
    # update connection status and mode to UI
    if (mode is not None):
        self.platform_mode.set(mode)
        if (mode == "MODE_DISABLED"):
            self.platform_mode_label.config(bg="#ff0000", fg="#ffffff")
        elif (mode == "MODE_ENABLE"):
            self.platform_mode_label.config(bg="#23ff23", fg="#000000")
    pass

def toggle_platform_mode():
    """Enable/disable platform operating mode"""
    if (self.MovingPlatform.get_is_enabled() == True):
        self.MovingPlatform.disable_platform()
        self.platform_enable_button.config(text="Enable Platform")
    else:
        self.MovingPlatform.enable_platform()
        self.platform_enable_button.config(text="Disable Platform")
    pass

def clear_platform_error():
    """Clear error"""
    self.MovingPlatform.clear_error()
    pass


def reset_platform():
    """Reset platform"""
    self.MovingPlatform.reset()
    pass