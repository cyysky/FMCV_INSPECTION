from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from tkinter.messagebox import askokcancel, showinfo, WARNING

from PIL import ImageTk, Image
import tkinter as tk
import cv2
import base64
import copy
import traceback
from datetime import datetime,timedelta

from FMCV.Ui.ScrollableFrm  import ScrollableFrame
from FMCV import Util
from FMCV.Ui.ImageViewFrm import ImageView
import numpy as np


operation_results_window = None
barcode = ""

def init(in_start):
    global start
    start = in_start
    #if in_start.Config.log_type in ( "PLEXUS","KAIFA"):
    start.sub("ui/results/show", new_window)
    start.sub("ui/results/set_barcode", set_barcode)
    
def set_barcode(in_barcode):
    global barcode
    
    barcode = in_barcode
    start.log.info(f"barcode {barcode}")

#https://pythonprogramming.altervista.org/tkinter-open-a-new-window-and-just-one/
def new_window():
    global window_opened_root
    global operation_results_window
    try:
        if window_opened_root.state() == "normal": window_opened_root.focus()
    except:
        #traceback.print_exc()
        start.log.info("Creating New OperationResultsWindow")
        window_opened_root = tk.Toplevel()
        operation_results_window = OperationResultsWindow(start, window_opened_root)

def refresh():
    global window_opened_root
    global operation_results_window
    try:
        operation_results_window.main_frame.refresh()
    except:
        pass
        #traceback.print_exc()
        #print("operation_results_window didnt opened")
        
 
class OperationResultsWindow:
    
    def __init__(self, start, root):
        self.start = start
        self.root = root
        root.geometry("1024x768+150+150")  
        self.main_frame = OperationResultsFrame(start, root)
        self.main_frame.pack(fill=BOTH, expand=True)
        self.root.attributes('-topmost',True)

class OperationResultsFrame(ttk.Frame):

    def __init__(self,start, parent, *args, **kwargs):
        #https://stackoverflow.com/questions/17466561/best-way-to-structure-a-tkinter-application/17470842
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.start = start
        self.current_roi = None
        
        # Main Section
        level_1_horizontal = ttk.PanedWindow(self,orient=HORIZONTAL)
        level_1_horizontal.pack(fill=BOTH, expand=True)
        #level_1_horizontal_frame = ScrollableFrame(level_1_horizontal)
        #control_frame = level_1_horizontal_frame.scrollable_frame
        
        # 4x Image View Start
        images_panedwindow = ttk.PanedWindow(self,orient=HORIZONTAL)
        images_panedwindow.pack(fill=BOTH, expand=True)
        
        left_vertical = ttk.PanedWindow(self,orient=VERTICAL)
        left_vertical.pack(fill=BOTH, expand=True)
        
        left_upper_horizontal = ttk.PanedWindow(self,orient=HORIZONTAL)
        left_upper_horizontal.pack(fill=BOTH, expand=True)
        
        frm_up = ttk.Frame(self)
        frm_up.pack(fill=BOTH, expand=True)
        self.roi_current_label = Label(frm_up, text = "N/A")
        self.roi_current_label.pack(fill=X)
        self.image_view_current = ImageView(frm_up, relief = GROOVE, borderwidth = 2)
        self.image_view_current.pack(fill=BOTH, expand=True)
        
        frm_up_right = ttk.Frame(self)
        frm_up_right.pack(fill=BOTH, expand=True)
        self.overall_result = Label(frm_up_right, text = "N/A", font=("Arial", 25))
        self.overall_result.pack(fill=BOTH, expand=True)
        
        left_upper_horizontal.add(frm_up, weight = 10)
        left_upper_horizontal.add(frm_up_right, weight = 4)
        
        frm_down = ttk.Frame(self)
        frm_down.pack(fill=BOTH, expand=True)
        self.roi_good_label = Label(frm_down, text = "Good template")
        self.roi_good_label.pack(fill=X)
        self.image_view_good = ImageView(frm_down, relief = GROOVE, borderwidth = 2)
        self.image_view_good.pack(fill=BOTH, expand=True)
        left_vertical.add(left_upper_horizontal, weight = 10)
        left_vertical.add(frm_down, weight = 10)

        images_panedwindow.add(left_vertical, weight = 5)
        
        right_vertical = ttk.PanedWindow(self,orient=VERTICAL)
        right_vertical.pack(fill=BOTH, expand=True)
        
        frm_up = ttk.Frame(self)
        frm_up.pack(fill=BOTH, expand=True)
        Label(frm_up, text = "Current View").pack(fill=X)
        self.image_view_roi = ImageView(frm_up,zoom_callback=self.update_result_roi, relief = GROOVE, borderwidth = 2)
        self.image_view_roi.pack(fill=BOTH, expand=True)
        frm_down = ttk.Frame(self)
        frm_down.pack(fill=BOTH, expand=True)
        self.place_holder_label = Label(frm_down, text = "Place Holder")
        self.place_holder_label.pack(fill=X)
        self.image_view_all = ImageView(frm_down, relief = GROOVE, borderwidth = 2)
        self.image_view_all.pack(fill=BOTH, expand=True)
        
        right_vertical.add(frm_up, weight = 10)
        right_vertical.add(frm_down, weight = 10)

        images_panedwindow.add(right_vertical, weight = 5)
        # 4x Image View End
        
        # Control Level
        control_level_horizontal = ttk.PanedWindow(self, orient=HORIZONTAL)
        control_level_horizontal.pack(fill=BOTH,expand=True)
        
        suggestion_vertical = ttk.PanedWindow(self, orient=VERTICAL)
        suggestion_vertical.pack(fill=BOTH,expand=True)
        
        control_frm = ttk.Frame(self)
        control_frm.pack(fill=BOTH,expand=True)
        
        # Control Section Start
        btn_refresh = Button(control_frm,text ="Refresh", command = self.refresh)
        btn_refresh.pack()
        self.results_label = Label(control_frm, text = 'N/A')
        self.results_label.pack()
        
        Label(control_frm, text = 'Results Filter').pack()
        self.result_type_cmb = result_type_cmb = ttk.Combobox(control_frm,state="readonly")
        result_type_cmb['values'] = ("FAIL","PASS","ALL")
        result_type_cmb.pack()
        result_type_cmb.current(0)
        result_type_cmb.bind("<<ComboboxSelected>>", self.roi_type_cmb_callback)
        
        self.control_1_lbl = Listbox(control_frm, exportselection=0)
        self.control_1_lbl.pack(fill=BOTH,expand=True)
        self.control_1_lbl.bind("<<ListboxSelect>>", self.control_1_lbl_callback)
        scrollbar = Scrollbar(self.control_1_lbl)
        scrollbar.pack(side = RIGHT, fill = BOTH)
        self.control_1_lbl.config(yscrollcommand = scrollbar.set)
        scrollbar.config(command = self.control_1_lbl.yview)
        
        self.show_btn = Button(control_frm,text ="Show")
        self.show_btn.pack()
        self.show_btn.configure(command=self.show_callback) 
       
        # Control Frame End           
        control_level_horizontal.add(control_frm, weight = 2)
        
        # Detail Frame Start
        #detail_scfrm = ScrollableFrame(self)
        #detail_scfrm.pack(fill=BOTH,expand=True)
        #detail_frm = detail_scfrm.scrollable_frame 
        detail_frm = ttk.Frame(self)
        detail_frm.pack(fill=BOTH,expand=True)
        
        Label(detail_frm,text ="Class suggestion").pack()
        
        frm = ttk.Frame(detail_frm)
        frm.pack(fill=X)
        btn_submit = Button(frm,text ="Submit", command = self.submit_suggestion_list)
        btn_submit.pack(side = RIGHT)
        btn_clear_list = Button(frm,text ="clear", command = self.clear_suggestion_list)
        btn_clear_list.pack(side = LEFT)
              
        Label(detail_frm,text ="Details").pack()
        
        self.detail_1_lbl = Listbox(detail_frm, exportselection=0)
        self.detail_1_lbl.pack(fill=BOTH,expand=True)
        self.detail_1_lbl.bind("<<ListboxSelect>>", self.control_1_lbl_callback)
        scrollbar = Scrollbar(self.detail_1_lbl)
        scrollbar.pack(side = RIGHT, fill = BOTH)
        self.detail_1_lbl.config(yscrollcommand = scrollbar.set)
        scrollbar.config(command = self.detail_1_lbl.yview)
        
        # Detail Frame End
        control_level_horizontal.add(detail_frm, weight = 2)
        
        # Suggestion Frame
        suggestion_frm = ttk.Frame(self)
        suggestion_frm.pack(fill=BOTH,expand=True)
        
        Label(suggestion_frm, text = 'Classes').pack()
        self.folder_cmb = folder_cmb = ttk.Combobox(suggestion_frm)
        folder_cmb['values'] = tuple(start.Profile.get_image_folders_list())
        folder_cmb.pack(fill=BOTH, expand=True)
        
        self.btn_suggestion = btn_suggestion = Button(suggestion_frm,text ="Class Suggestion")
        btn_suggestion.pack()
        btn_suggestion.configure(command=self.roi_class_suggestion)
        
        #self.btn_open_folder = Button(suggestion_frm,text ="Open Folder", command = lambda:start.ActionUi.open_folder(folder_cmb.get()))
        #self.btn_open_folder.pack()
        
        suggestion_vertical.add(control_level_horizontal, weight = 100)
        suggestion_vertical.add(suggestion_frm, weight = 1)
        
        
        # Main Section Left - Right
        level_1_horizontal.add(suggestion_vertical, weight = 2)
        level_1_horizontal.add(images_panedwindow, weight = 9)
        
        self.statusbar = Label(self, text="Operation Results View", bd=1, relief=SUNKEN, anchor=W)
        self.statusbar.pack(side=BOTTOM, fill=X)
        
        self.selected = (0,)
        
        #self.parent.bind("<Configure>", self.configure_event)
        
        parent.after(700, self.refresh)  
        
    def configure_event(self,event):
        self.show_selected()

    
    def select_combobox_by_text(self,combobox, selected_text):
        if selected_text in combobox['values']:
            combobox.set(selected_text)
        else:
            self.start.log.debug("Combobox option not found!")
        
    def roi_class_suggestion(self):
        if start.Users.login_user():
            selection = self.control_1_lbl.curselection()
            if selection:
                index = selection[0]
                data = self.control_1_lbl.get(index)
                R = self.start.Main.results
                n = data.split(",")
                #print(f"data splited is {n}")
                srn_n = int(n[0])-1
                step_n = int(n[1])-1
                roi_n = int(n[2])-1
                self.current_roi = current_roi = R[srn_n][step_n][roi_n]
                
                if current_roi.get("type") in ("AI","CNN"):

                    if self.folder_cmb.get() != "" :
                        # find match to suggested in list
                        matched = -1
                        values = self.detail_1_lbl.get(0, END)
                        for n, v in enumerate(values):
                            d = v.split(",")
                            d_srn_n = int(d[0])-1
                            d_step_n = int(d[1])-1
                            d_roi_n = int(d[2])-1
                            
                            if srn_n == d_srn_n and step_n == d_step_n and roi_n == d_roi_n:
                                matched = n
                                
                        # check if exists        
                        if matched > -1: 
                            #https://stackoverflow.com/questions/36086474/python-tkinter-update-scrolled-listbox-wandering-scroll-position
                            vw = self.detail_1_lbl.yview()
                            display_str = f"{srn_n+1},{step_n+1},{roi_n+1},{self.folder_cmb.get()}"
                            self.detail_1_lbl.delete(matched)
                            self.detail_1_lbl.insert(matched-1,display_str)
                            self.detail_1_lbl.yview_moveto(vw[0])
                        else:
                            display_str = f"{srn_n+1},{step_n+1},{roi_n+1},{self.folder_cmb.get()}"
                            self.detail_1_lbl.insert(END, display_str)
                    else:
                        #messagebox.askokcancel("Class Suggestion", "Please select a class suggest")
                        date_time_str = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
                        self.statusbar.config(text = f"Please select a class suggest {date_time_str}")
                else:
                    #messagebox.askokcancel("Class Suggestion", "Current roi type is not CNN")
                    date_time_str = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
                    self.statusbar.config(text = f"Current roi type is not AI {date_time_str}")
        
    def clear_suggestion_list(self):
        self.detail_1_lbl.delete(0,END)
        
    def submit_suggestion_list(self):
        if start.Users.login_user():
            log_datetime = datetime.now().strftime("%Y%m%d_%H%M%S") #https://www.w3schools.com/python/python_datetime.asp
        
            R = self.start.Main.results
            values = self.detail_1_lbl.get(0, END)
            for n, v in enumerate(values):
                d = v.split(",")
                d_srn_n = int(d[0])-1
                d_step_n = int(d[1])-1
                d_roi_n = int(d[2])-1
                d_class = d[3]
                result_roi = R[d_srn_n][d_step_n][d_roi_n]
                img = result_roi.get('result_image')
                if img is not None:
                    file_name = f"{barcode}_{d_class}_{start.Users.username}_{log_datetime}_{datetime.now().strftime('%f')}.png"
                    
                    self.start.Profile.write_ai_suggestion_image(d_class,file_name,img)
                    
                    images_path = self.start.Config.images_path
                    if str(images_path) != '.':
                        #barcode = self.start.Main.barcode
                        images_path_2 = images_path / "Operator" / str(barcode +"_"+log_datetime +"_"+start.Users.username) / d_class 
                        images_path_2.mkdir(parents=True, exist_ok=True)
                        
                        cv2.imwrite(str(images_path_2 / file_name), img)
                        
                        #pth = self.start.Profile.write_image(d_class, img)
                        self.statusbar.config(text = f'Result Image Saved to {str(images_path_2 / file_name)}')
                        result_roi['result_class'] = d_class
                        result_roi['result_score'] = -999
                        result_roi['PASS'] = True
                        result_roi['edit_user'] = start.Users.username
            self.start.Main.barcode = barcode
            self.start.Log.write_log()
            self.start.Main.barcode = ""
            self.detail_1_lbl.delete(0,END)
        
        
    def save_result_roi_image(self):
        if self.current_roi is not None:
            img = self.current_roi.get('result_image')
            if img is not None:
                #self.start.Profile.create_image_folder(self.folder_cmb.get())
                pth = self.start.Profile.write_image(self.folder_cmb.get(),img)
                print(self.folder_cmb.get())
                self.start.MainUi.write(f'Result Image Saved to {pth}')
        self.folder_cmb['values'] = tuple(start.Profile.get_image_folders_list())
        self.folder_cmb.pack()
        
    def roi_type_cmb_callback(self,*args):
        self.refresh()

    def control_1_lbl_callback(self,*args):
        self.selected = self.control_1_lbl.curselection()
        #print(type(self.selected))
        self.show_selected()
        
    def show_callback(self,*args):
        self.show_selected()
    
    def show_selected(self):
        # check selection
        #self.selected = selected = self.control_1_lbl.curselection()
        selected = self.selected
        if not selected:
            return 

        # select last item
        if selected[0] < self.control_1_lbl.size():        
            self.control_1_lbl.select_clear(0, tk.END)
            self.control_1_lbl.select_set(selected[0])    
            self.control_1_lbl.activate(selected[0])
            self.control_1_lbl.see(selected[0]) #https://stackoverflow.com/questions/10155153/how-to-scroll-to-a-selected-item-in-a-scrolledlistbox-in-python
        elif self.control_1_lbl.size() > 0:
            self.control_1_lbl.select_clear(0, tk.END)
            self.control_1_lbl.select_set(0)    
            self.control_1_lbl.activate(0)
            self.control_1_lbl.see(0)
                
        # Reset Ui text and color
        self.roi_current_label.config(bg="SystemButtonFace",text="N/A")
        self.image_view_current.viewer.config(highlightbackground="gray")
        self.image_view_current.clear()
        self.image_view_good.clear()
        self.image_view_roi.clear()
        self.image_view_roi.viewer.delete("all")
        self.image_view_all.clear() 
        
        # Display selected information
        data = self.control_1_lbl.get(selected[0])
        if not data:
            return
        R = self.start.Main.results
        n = data.split(",")
        #print(f"data splited is {n}")
        self.current_roi = current_roi = R[int(n[0])-1][int(n[1])-1][int(n[2])-1]
        
        self.select_combobox_by_text(self.folder_cmb,current_roi.get('class'))
        
        if current_roi.get('result_image') is None:
            return
            
        if current_roi.get('result_blended_heatmap') is not None:
            self.image_view_current.set_image(current_roi.get('result_blended_heatmap'),reset_scale = True)
        else:
            self.image_view_current.set_image(current_roi.get('result_image'),reset_scale = True)

        self.image_view_current.viewer.config(highlightthickness=1)
        self.image_view_current.viewer.config(highlightbackground="gray")
        roi_pass = current_roi.get('PASS')
        
        if roi_pass is True:
            self.roi_current_label.config(bg='green',text="GOOD")
            if current_roi.get("type") in ("AI","CNN") and current_roi.get('result_score') is not None:
                try:
                    self.roi_current_label.config(bg='green',text=f"GOOD ({current_roi.get('result_class')} {round(current_roi.get('result_score'),4)})")
                except:
                    traceback.print_exc()                
            self.image_view_current.viewer.config(highlightbackground='green')
        else:
            if current_roi.get("type") in ("AI","CNN") and current_roi.get('result_score') is not None:
                try:
                    self.roi_current_label.config(bg='red',text=f"NG ({current_roi.get('result_class')} {round(current_roi.get('result_score'),4)})")
                except:
                    traceback.print_exc()
            else:
                self.roi_current_label.config(bg='red',text="NG")
                 
            self.image_view_current.viewer.config(highlightbackground='red')
        
        # get good template image
        if current_roi.get("type") in ("AI","CNN") and current_roi.get('result_score') is not None:
            try:
                self.roi_good_label.config(text=f"Good template ({current_roi.get('class')} {round(current_roi.get('minimal'),4)})")
            except:
                traceback.print_exc()
        else:
            self.roi_good_label.config(text=f"Good template")
        
        self.image_view_good.set_image(current_roi.get('img'),reset_scale = True)
        
        # get captured image
        try:
            self.image_view_roi.viewer.delete("all")
            self.image_view_roi.set_image(self.start.Main.result_frame[int(n[0])-1][int(n[1])-1][current_roi["rotate"]],reset_scale = True)
            if current_roi.get("source_image") is not None:
                self.image_view_roi.set_image(current_roi.get("source_image"),reset_scale = True)
            roi_pass = current_roi.get('PASS')
            if roi_pass is True:
                color = 'green'
            else:
                color = 'red'
            
            scale = self.image_view_roi.get_scale()
            
            off_x = 0
            off_y = 0
            
            roi = current_roi
            if roi.get("offset_x") is not None:
                off_x = roi["offset_x"]
                
            if roi.get("offset_y") is not None:
                off_y = roi["offset_y"]
                
            self.image_view_roi.viewer.create_rectangle((roi['x1']+off_x) * scale, (roi['y1']+off_y) * scale, (roi['x2']+off_x) * scale, (roi['y2']+off_y) * scale , outline=color)
        except:
            traceback.print_exc()
        
        try:
            if self.start.Main.result_frame[int(n[0])-1][int(n[1])-1][current_roi["rotate"]] is not None:
                self.image_view_all.set_image(self.start.Main.result_frame[int(n[0])-1][int(n[1])-1][current_roi["rotate"]],reset_scale = True)
            if current_roi.get("source_image") is not None:
                self.image_view_all.set_image(current_roi.get("source_image"),reset_scale = True)
        except:
            traceback.print_exc()
            
        if current_roi.get("ano_result_image") is not None:
            self.image_view_all.viewer.delete("all")
            if isinstance(current_roi.get("ano_result_image"),np.ndarray):
                self.image_view_all.set_image(current_roi.get("ano_result_image"),reset_scale = True)

        if current_roi.get("2D_result_image") is not None:
            self.image_view_all.viewer.delete("all")
            if isinstance(current_roi.get("2D_result_image"),np.ndarray):
                self.image_view_all.set_image(current_roi.get("2D_result_image"),reset_scale = True)

    def update_result_roi(self):
        selected = self.selected
        if not selected:
            return 
        # Display selected information
        data = self.control_1_lbl.get(selected[0])
        R = self.start.Main.results
        n = data.split(",")
        #print(f"data splited is {n}")
        self.current_roi = current_roi = R[int(n[0])-1][int(n[1])-1][int(n[2])-1]
        
        self.image_view_roi.clear_all_except_picture()
        roi_pass = current_roi.get('PASS')
        if roi_pass is True:
            color = 'green'
        else:
            color = 'red'
        
        scale = self.image_view_roi.get_scale()
        
        off_x = 0
        off_y = 0
        
        roi = current_roi
        if roi.get("offset_x") is not None:
            off_x = roi["offset_x"]
            
        if roi.get("offset_y") is not None:
            off_y = roi["offset_y"]
            
        self.image_view_roi.viewer.create_rectangle((roi['x1']+off_x) * scale, (roi['y1']+off_y) * scale, (roi['x2']+off_x) * scale, (roi['y2']+off_y) * scale , outline=color)

                    
    def refresh(self,*args):
        self.image_view_current.viewer.config(highlightbackground="gray")
        self.image_view_current.clear()
        self.image_view_good.clear()
        self.image_view_roi.clear()
        self.image_view_roi.viewer.delete("all")
        self.image_view_all.clear()        
        self.roi_current_label.config(bg="SystemButtonFace",text="N/A")
        
        self.results_label.config(text = "N/A")
        self.control_1_lbl.delete(0,END)
        
        R = self.start.Main.results
        already_checked = 0
        total_checked = 0

        for src_n, src in enumerate(R):
            for pos_n, pos in enumerate(R[src_n]):
                for roi_n, roi in enumerate(R[src_n][pos_n]):
                    try:                        
                        result_type = self.result_type_cmb.get()
                        #print(f"result {roi.get('PASS')} , result_type is {result_type}")
                        if result_type == 'FAIL':
                            if roi.get('PASS') is False:
                                str_display = f"{src_n+1},{pos_n+1},{roi_n+1},{roi['name']}"
                                self.control_1_lbl.insert(END, str_display)

                        if result_type == 'PASS':
                            if roi.get('PASS') is True:
                                str_display = f"{src_n+1},{pos_n+1},{roi_n+1},{roi['name']}"
                                self.control_1_lbl.insert(END, str_display)

                        if result_type == 'ALL':
                            str_display = f"{src_n+1},{pos_n+1},{roi_n+1},{roi['name']}"
                            self.control_1_lbl.insert(END, str_display)

                        if roi.get('PASS') is not None:
                            already_checked = already_checked + 1
                    except:
                        traceback.print_exc()
                    total_checked = total_checked + 1
                    
        self.results_label.config(text = f"Total Checked {already_checked}/{total_checked}")

        # #print(self.control_1_lbl.size())
        # if self.control_1_lbl.size() > 0:
            # if not self.selected:
                # self.control_1_lbl.select_clear(0, tk.END)
                # self.control_1_lbl.select_set(0)    
                # self.control_1_lbl.activate(0)
                # self.control_1_lbl.see(0)
                # #self.control_1_lbl.select_set(self.control_1_lbl.index("end"))
                
        self.show_selected()
            
        if already_checked == total_checked:   
            if self.start.Main.is_overall_pass():
                self.overall_result.config(bg='green',text=f"Overall \nGOOD")
            else:
                self.overall_result.config(bg='red',text=f"Overall \nNG")
        else :
            self.overall_result.config(bg="SystemButtonFace",text=f"Wait")