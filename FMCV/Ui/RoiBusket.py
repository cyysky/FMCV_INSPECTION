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
import datetime

try:
    from FMCV.Ui.ScrollableFrm  import ScrollableFrame
    from FMCV import Util
    from FMCV.Ui.ImageViewFrm import ImageView
except:
    pass
import numpy as np

start = None

def init(in_start):
    global start
    start = in_start
    #if in_start.Config.log_type in ( "PLEXUS","KAIFA"):
    start.sub("ui/roi_busket/show", new_window)
    
#https://pythonprogramming.altervista.org/tkinter-open-a-new-window-and-just-one/
def new_window():
    global window_opened_root
    global roi_busket_window
    try:
        if window_opened_root.state() == "normal": window_opened_root.focus()
    except:
        start.log.info("Creating New RoiBusketWindow")
        window_opened_root = tk.Toplevel()
        roi_busket_window = RoiBusketWindow(start, window_opened_root)

def refresh():
    global window_opened_root
    global roi_busket_window
    try:
        roi_busket_window.main_frame.refresh()
    except:
        start.log.debug(traceback.format_exc())
        
 
class RoiBusketWindow:
    def __init__(self, start, root):
        self.start = start
        self.root = root
        root.title("ROI Busket")
        root.geometry("300x400+150+150")  
        self.main_frame = RoiBusketFrame(start, root)
        self.main_frame.pack()
        self.root.attributes('-topmost',True)


class RoiBusketFrame(ttk.Frame):
    def __init__(self,start, parent, *args, **kwargs):
        #https://stackoverflow.com/questions/17466561/best-way-to-structure-a-tkinter-application/17470842
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.start = start
        self.current_roi = None
        self.master = parent

        self.listbox = tk.Listbox(self.master, selectmode=tk.MULTIPLE)
        self.add_button = tk.Button(self.master, text="Add from Current Selected", command=self.add_rectangle)
        self.delete_button = tk.Button(self.master, text="Delete", command=self.delete_rectangle)
        self.edit_button = tk.Button(self.master, text="Edit", command=self.edit_rectangle)
        self.paste_button = tk.Button(self.master, text="Paste", command=self.paste_rectangle)

        self.rois = []
        self.rect_id = 0

        self.listbox.pack(fill = BOTH, expand=True)
        self.add_button.pack()
        self.delete_button.pack()
        self.edit_button.pack()
        self.paste_button.pack()
        
    def add_rectangle(self):
        if not self.start.Users.login_admin():
            return
        
        ret, roi = self.start.Profile.get_selected_roi()
        if not ret:
            self.log.info("No ROI is selected")
            return
                    
        self.rois.append(roi)
        self.listbox.insert(tk.END, roi['name'])

        
    def paste_rectangle(self):
        if not self.start.Users.login_admin():
            return
            
        selection = self.listbox.curselection()
        if selection:
            for index in selection:
                self.start.Profile.paste_roi(self.rois[index])
            
    def create_rectangle(self):
        self.rect_id += 1
        rect = self.canvas.create_rectangle(50, 50, 100, 100)
        self.rectangles[self.rect_id] = rect
        self.listbox.insert(tk.END, f'Rectangle {self.rect_id}')

    def delete_rectangle(self):
        selection = self.listbox.curselection()
        if selection:
            # Iterate over the selected indices in reverse order
            for index in reversed(selection):
                self.rois.pop(index)
                self.listbox.delete(index)
            
    def edit_rectangle(self):
        selected = self.listbox.curselection()
        if len(selected) != 1:
            messagebox.showinfo("Error", "Select a single rectangle to edit.")
            return
        roi = self.rois[selected[0]]
        self.start.Profile.select_roi(roi)


    def copy_rectangle(self):
        selected = self.listbox.curselection()
        for index in selected:
            rect_id = index + 1
            coords = self.canvas.coords(self.rectangles[rect_id])
            new_rect = self.canvas.create_rectangle(*coords)
            self.rect_id += 1
            self.rectangles[self.rect_id] = new_rect
            self.listbox.insert(tk.END, f'Rectangle {self.rect_id}')
      
if __name__ == "__main__":
    root = tk.Tk()
    #window_opened_root = tk.Toplevel()
    window_opened_root = root
    operation_results_window = RoiBusketWindow(start, window_opened_root)
    root.mainloop()