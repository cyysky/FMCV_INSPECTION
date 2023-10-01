from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image
import cv2
from FMCV.Ui.ScrollableFrm  import ScrollableFrame
from FMCV import Util
from FMCV.Ui.PlatformSetting import PlatformSettingFrame
from FMCV.Ui.OperatorTop import OperatorWindow
from FMCV import Logging
import os
class ControlFrame(ttk.Frame):

    def __init__(self, start, parent, *args, **kwargs):
        #https://stackoverflow.com/questions/17466561/best-way-to-structure-a-tkinter-application/17470842
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.start = start
        
        sf = ScrollableFrame(self)
        sf.pack(fill = BOTH, expand=True)
        frm = sf.scrollable_frame
        
        self.img = ImageTk.PhotoImage(Image.open(os.path.join("FMCV","Ui","fmcv-square-128.png")))


        # Create a Label Widget to display the text or Image
        label = Label(frm, image = self.img)
        label.pack()

        Label(frm, text = "FREE VERSION").pack()
            
        Label(frm, text = 'Profile').pack()
        
        self.cmb_profile = cmb_profile = ttk.Combobox(frm,state="readonly")
        cmb_profile.pack()
        
        Label(frm, text = 'Sources').pack()
       
        self.btn_add_source = btn_add_source = Button(frm, text ="Insert Sources")
        btn_add_source.pack(side=TOP)        
        self.btn_remove_source = btn_remove_source = Button(frm, text ="Remove Sources")
        btn_remove_source.pack(side=TOP)
        #cmb_cam.bind("<<ComboboxSelected>>", cmb_callback)
        self.cmb_cam = cmb_cam = ttk.Combobox(frm,state="readonly")
        cmb_cam.pack()
        
        Label(frm, text = 'Steps').pack()
        step_button_frame = ttk.Frame(frm)
        step_button_frame.pack()
        
        fl = ttk.Frame(step_button_frame)
        fl.pack(side="right")
        
        self.btn_up_step = btn_up_step = Button(fl, text =" Up ")
        btn_up_step.pack()
        self.btn_down_step = btn_down_step = Button(fl, text ="Down")
        btn_down_step.pack()
        
        fr = ttk.Frame(step_button_frame)
        fr.pack(side="left")
        
        self.btn_add_step_above = btn_add_step_above = Button(fr, text ="Insert Step")
        btn_add_step_above.pack()


        self.btn_add_step = btn_add_step = Button(fr, text ="Add Step")
        btn_add_step.pack()
                
        self.btn_remove_step = btn_remove_step = Button(fr, text ="Remove Step")
        btn_remove_step.pack()

        self.btn_duplicate_step = btn_duplicate_step = Button(fr, text ="Duplicate Step")
        btn_duplicate_step.pack()
        
        
        
        self.cmb = cmb = ttk.Combobox(frm,state="readonly")
        cmb.pack()
        #cmb.bind("<<ComboboxSelected>>", cmb_callback)
        
        
        Label(frm, text = 'ROI Name').pack()        
        self.roi_entry = roi_entry = Entry(frm)
        roi_entry.pack()
        
        
        
        move_all_frame = ttk.Frame(frm)
        
        ttk.Label(move_all_frame, text="Move all ROI").pack(side=LEFT)
        self.move_all_var = BooleanVar()
        self.move_all_var.set(False)
        self.move_all_checkbox = ttk.Checkbutton(move_all_frame, variable=self.move_all_var, command=lambda: Logging.debug("move all ",self.move_all_var.get()))
        self.move_all_checkbox.pack(side=RIGHT)
        
        move_all_frame.pack()
        
        
        
        ff = ttk.Frame(frm)
        ff.pack()
        self.btn_add_roi = btn_add_roi = Button(ff, text ="Add Roi")
        btn_add_roi.pack(side=LEFT)
        self.btn_roi_up = btn_roi_up = Button(ff, text ="Up")
        btn_roi_up.pack(side=LEFT)
        self.btn_roi_down = btn_roi_down = Button(ff, text ="Down")
        btn_roi_down.pack(side=LEFT)

        frm_lbl = Frame(frm)
        frm_lbl.pack()
        self.Lb1 = Lb1 = Listbox(frm_lbl,exportselection=0)#,selectmode=MULTIPLE)#selectmode=SINGLE
        Lb1.pack(side=LEFT)
        #Lb1.bind("<<ListboxSelect>>", Lbl_callback)
        scrollbar = Scrollbar(frm_lbl)
        scrollbar.pack(side = RIGHT, fill = BOTH)
        Lb1.config(yscrollcommand = scrollbar.set)
        scrollbar.config(command = Lb1.yview)
        
        
        ff = ttk.Frame(frm)
        ff.pack()
        
        fl = ttk.Frame(ff)
        fl.pack(side=LEFT)
 
        self.btn_rename_roi = btn_rename_roi = Button(fl,text ="Rename Roi")
        btn_rename_roi.pack(side=LEFT)
        self.btn_remove_roi = btn_remove_roi = Button(fl,text ="Remove Roi")
        btn_remove_roi.pack(side=RIGHT)
        
        fr = ttk.Frame(ff)
        fr.pack(side=RIGHT)

        Label(frm, text = 'Settings').pack()
        
        save_btn = Button(frm, text ="Save Setting", command = start.ActionUi.save_profile)
        save_btn.pack(side=TOP)
        
        button_1 = Button(frm, text ="Reset Detection Step", command = start.ActionUi.reset_detection_step)
        button_1.pack(side=TOP)
        Label(frm, text = 'Barcode').pack()
        self.barcode_entry = barcode_entry = Entry(frm)
        barcode_entry.pack()
        
        Label(frm, text = 'Detections & Robot').pack()
        button_1 = Button(frm, text ="Detect", command = start.ActionUi.detect)
        button_1.pack(side=TOP)
        button_1 = Button(frm, text ="Detect Selected Step", command = start.ActionUi.detect_current_step)
        button_1.pack(side=TOP)
        self.steps_lbl = Label(frm, text = '')
        self.steps_lbl.pack()
        
        if start.Config.config.HOST.broadcast_port > 1024 and start.Config.config.HOST.broadcast_port < 65535:
            button_1 = Button(frm, text ="Start Robot", command = lambda:start.host.broadcast_message(start.Profile.name))
            button_1.pack(side=TOP)
        
        button_1 = Button(frm, text ="Skip Step", command = start.ActionUi.skip_step)
        button_1.pack(side=TOP)
        button_1 = Button(frm, text ="Robot Next Step", command = start.ActionUi.go_next)
        button_1.pack(side=TOP)
            
        # Platform setting button
        if start.Config.platform_model != "NONE":
            labelframe_platform = LabelFrame(frm, bd=3, text="Platform")
            labelframe_platform.pack(fill='x', ipadx=10, ipady=10, padx=5, pady=5)
            button_platform_setting = Button(labelframe_platform, text="Setting")
            button_platform_setting.bind("<Button>", lambda e: PlatformSettingFrame(start, start.MainUi.top))
            button_platform_setting.pack(fill='x', ipadx=5, ipady=5, padx=5, pady=5)
        
        #Open new window
        #button_1 = Button(frm, text ="New Window", command = start.ActionUi.go_next)
        #button_1.pack(side=TOP)
        #button_1.bind("<Button>",lambda e: OperatorWindow(start,start.MainUi.top))
        #button_1.pack(side=TOP)
        
        
        if start.Config.mode_name != "ENGINEER":
            btn_add_roi["state"] = DISABLED
            btn_remove_roi["state"] = DISABLED
            btn_rename_roi["state"] = DISABLED
            btn_add_step["state"] = DISABLED
            btn_add_step_above["state"] = DISABLED
            btn_remove_step["state"] = DISABLED
            btn_remove_roi["state"] = DISABLED
            btn_add_source["state"] = DISABLED
            btn_remove_source["state"] = DISABLED
            save_btn["state"] = DISABLED
            
