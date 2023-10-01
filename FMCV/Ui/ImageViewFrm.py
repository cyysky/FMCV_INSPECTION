from tkinter import *
from PIL import ImageTk, Image
import cv2
import traceback
import sys
import numpy as np

import inspect
 
from FMCV.Cv import Cv
 
class ImageView(Frame):

    def __init__(self, parent, zoom_callback=None, *args, **kwargs):
        #https://stackoverflow.com/questions/17466561/best-way-to-structure-a-tkinter-application/17470842
        Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.zoom_callback = zoom_callback
        
        self.hbar = hbar = Scrollbar(self,orient=HORIZONTAL, width=20)
        hbar.pack(side=BOTTOM,fill=X)
        
        self.vbar = vbar = Scrollbar(self,orient=VERTICAL, width=20)
        vbar.pack(side=RIGHT,fill=Y)
        
        self.viewer = Canvas(self,width=100,height=300,scrollregion=(0,0,640,480), highlightthickness=2)

        vbar.config(command=self.viewer.yview)      
        hbar.config(command=self.viewer.xview)  
        
        self.viewer.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.viewer.pack(expand=True,fill=BOTH)
        
        self.id_picture = -1
        self.image = None
        self.scale = 1
        self.rotate = ""
        self.scale_by = "fit"
        
        # Set a bind on the mouse wheel scroll
        self.viewer.bind("<MouseWheel>", self.mouse_wheel)
        
        # Remember the default color:
        self.default_color = self.viewer.cget('highlightbackground')

    def set_border_color(self,color):
        #color = "red"
        if color == "default":
            color = self.default_color
        self.viewer.configure(highlightbackground=color)

    def view_to_image_position(self,x1,x2,y1,y2):
        x1 = int(self.viewer.canvasx(x1) / self.scale)
        x2 = int(self.viewer.canvasx(x2) / self.scale)
        y1 = int(self.viewer.canvasy(y1) / self.scale)
        y2 = int(self.viewer.canvasy(y2) / self.scale)
        return x1,x2,y1,y2
        
    def get_image(self):
        if self.image is None: 
            return False, None
            
        image = Cv.get_rotate_pil(self.rotate, self.image)
        
        # Convert PIL Image to NumPy array (image in RGB format)
        cv2_image = np.array(image)

        # Convert RGB to BGR 
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)
        return True, cv2_image

    def set_rotate(self, rotate_name):
        self.rotate = rotate_name

    def refresh_image(self):    
        self.update_image(self.image,self.scale)

    def mouse_wheel(self, event):
        # check for image existence and size before resizing
        if self.image is None:
            return
            
        image_width, image_height = self.image.size
        
        # Windows OS uses event.delta. Others may use event.num.
        last_scale = self.scale
        delta = event.delta / 120
        delta = delta * 0.07  # Change this value as you need for zoom sensitivity.
        
        # Check if reached max zoom scale
        MAX_PIXEL = 14000
        MIN_PIXEL = 30
        
        scale = self.scale + delta
        
        if scale * image_width > MAX_PIXEL or scale * image_height > MAX_PIXEL:
            print(f"Max pixel {MAX_PIXEL} reached")
            return 
            
        if scale * image_width < MIN_PIXEL or scale * image_height < MIN_PIXEL:
            print(f"Min pixel {MIN_PIXEL} reached")
            return 
            
        self.scale += delta
        
        # Get current canvas size
        canvas_view_width = self.viewer.winfo_width() - self.vbar.winfo_width()
        canvas_view_height = self.viewer.winfo_height() - self.hbar.winfo_height()
        #print(f"Canvas View Width {canvas_view_width} height {canvas_view_height}")
        
        # Get current view window size to image pixel
        view_width = self.viewer.canvasx(canvas_view_width) - self.viewer.canvasx(0)
        view_height = self.viewer.canvasy(canvas_view_height) - self.viewer.canvasy(0)
        
        # Get current mouse position
        mouse_position_ratio_x = (self.viewer.canvasx(event.x) - self.viewer.canvasx(0)) / view_width
        mouse_position_ratio_y = (self.viewer.canvasy(event.y) - self.viewer.canvasy(0)) / view_height
        #print(f"Mouse View Ratio x {mouse_position_ratio_x} y {mouse_position_ratio_y}")
        
        mouse_image_x = self.viewer.canvasx(event.x) / last_scale
        mouse_image_y = self.viewer.canvasy(event.y) / last_scale
        #print(f"Mouse Image x {mouse_image_x} y {mouse_image_y}")
        
        # redraw the image with the new scale
        self.update_image(self.image, self.scale)
        
        # Updated canvas image width to scale
        scaled_image_width, scaled_image_height = image_width*self.scale , image_height*self.scale
        
        # Updated mouse image width to scale
        mouse_image_x = mouse_image_x * self.scale
        mouse_image_y = mouse_image_y * self.scale
        #print(f"View Image center x {view_image_center_x} y {view_image_center_y}")
        
        # scroll the canvas to mouse position
        self.viewer.xview_moveto(((mouse_image_x - view_width * mouse_position_ratio_x)/scaled_image_width))
        self.viewer.yview_moveto(((mouse_image_y - view_height * mouse_position_ratio_y)/scaled_image_height))
    
    def get_scale(self):
        return self.scale
    
    def reset_scale(self):
        if self.image is not None: 
            image_width, image_height = self.image.size

            # Calculate scale for width and height
            width_scale = self.viewer.winfo_width() / image_width
            height_scale = self.viewer.winfo_height() / image_height

            # Choose the smaller scale to maintain aspect ratio
            self.scale = min(width_scale, height_scale)
            self.scale = self.scale * 0.985
        return self.scale
        
    def update_image(self, image:Image, scale):
        image_width,image_height = image.size
         
        width = int(image_width*scale)
        height = int(image_height*scale)
        
        if width < 1 or height < 1:
            return 
        
        if self.rotate != "":
            image = Cv.get_rotate_pil(self.rotate, image)
            image_width,image_height = image.size
            width = int(image_width*scale)
            height = int(image_height*scale)
            
        if not scale == 1:
            tk_image = image.resize((width,height))
        else:
            tk_image = image
  
        if self.id_picture > -1:
            self.viewer.image = None
            self.viewer.delete(self.id_picture)
  
        self.viewer.image = ImageTk.PhotoImage(tk_image)
        self.id_picture = self.viewer.create_image(0, 0, image=self.viewer.image, anchor='nw')
        self.viewer.tag_lower(self.id_picture)
        self.viewer.config(scrollregion=self.viewer.bbox('all'))
        
        # Once the zoom is complete, call the callback function (if one was provided)
        if self.zoom_callback is not None:
            self.zoom_callback()  # You could pass any information you want here

    def set_image(self,image : np.ndarray, reset_scale = False):
        try:
            if image is None:
                print("set_image image is None")
                return
                
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.image = image = Image.fromarray(img)
            
            if reset_scale or self.scale == 1:
                self.reset_scale()
            
            image_width,image_height = image.size
            width = int(image_width*self.scale)
            height = int(image_height*self.scale)
            
            if self.rotate != "":
                image = Cv.get_rotate_pil(self.rotate, image)
                image_width,image_height = image.size
                width = int(image_width*self.scale)
                height = int(image_height*self.scale)
            
            if self.id_picture > -1:
                self.viewer.image = None
                del self.viewer.image
                self.viewer.delete(self.id_picture)

            if width < 1 or height < 1:
                return 
            
            if not self.scale == 1:
                tk_image = image.resize((width,height))
            else:
                tk_image = image
                
            self.viewer.image = ImageTk.PhotoImage(tk_image)
            self.id_picture = self.viewer.create_image(0, 0, image=self.viewer.image, anchor='nw')
            self.viewer.tag_lower(self.id_picture)
            self.viewer.config(scrollregion=self.viewer.bbox('all'))
        except:
            #https://stackoverflow.com/questions/2654113/how-to-get-the-callers-method-name-in-the-called-method
            curframe = inspect.currentframe()
            calframe = inspect.getouterframes(curframe, 2)
            print('caller name:', calframe[1][3])
            traceback.print_exc()
            #print(sys.exc_info()[2])

    def lower_image_in_viewer(self):
        if self.id_picture > -1:    
            self.viewer.tag_lower(self.id_picture)
            
    def clear(self):
        if self.id_picture > -1:
            self.viewer.image = None
            self.viewer.delete(self.id_picture)
            
    def clear_all_except_picture(self):
        # Loop through all items on the canvas
        for item in self.viewer.find_all():
            # If the item's ID is not the one to keep, delete it
            if item != self.id_picture:
                self.viewer.delete(item)
                
