from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image
import cv2
        
class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        #https://stackoverflow.com/questions/17466561/best-way-to-structure-a-tkinter-application/17470842
        ttk.Frame.__init__(self, container, *args, **kwargs)
        self.parent = container
        
        canvas = Canvas(self,width=1,height=1,scrollregion=(0,0,1,1))
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0),window=self.scrollable_frame, anchor="nw")

        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        