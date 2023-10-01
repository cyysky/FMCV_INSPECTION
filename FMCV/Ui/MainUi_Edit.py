import copy
from FMCV import Logging
from FMCV.Ui import MainUi as M
from tkinter import messagebox
from tkinter import *
self = M.self

def set_text(entry_widget,text):
    entry_widget.delete(0,END)
    entry_widget.insert(0,text)

def make_roi_square(*args):
    if not self.Users.login_admin():
        return
    ret, roi = self.Profile.get_selected_roi()
    if ret:
        width = roi['x2'] - roi['x1']
        roi['y2'] = roi['y1'] + width
        self.MainUi.refresh()

def move_roi_center(*args):
    if not self.Users.login_admin():
        return
        
    ret, roi = self.Profile.get_selected_roi()
    if not ret:
        return
    
    h, w = list(self.Cam.get_image().values())[self.MainUi.cam_pos].shape[:2]
    
    rh,rw = roi['img'].shape[:2]
    roi['x1'] = 0
    roi['x2'] = rw
    roi['y1'] = 0
    roi['y2'] = rh
    
    roi_center_x = (roi["x1"] + roi["x2"]) / 2
    roi_center_y = (roi["y1"] + roi["y2"]) / 2
    
    image_center_x = w / 2
    image_center_y = h / 2
    
    shift_x = image_center_x - roi_center_x
    shift_y = image_center_y - roi_center_y
    print(shift_x,shift_y)

    roi['x1'] =  int(roi["x1"] + int(shift_x))
    roi['x2'] =  int(roi["x2"] + int(shift_x))
    roi['y1'] =  int(roi["y1"] + int(shift_y))
    roi['y2'] =  int(roi["y2"] + int(shift_y))

    self.MainUi.refresh()
    
def add_roi():
    print(M.roi_entry.get())
    if M.roi_entry.get() == "":
        messagebox.showerror(title="Please enter roi name", message="Please enter roi name")
    else:   
        x1,x2,y1,y2 = self.start.MainUi.view.view_to_image_position(10,100,10,100)
        self.Profile.add_roi(x1=x1,x2=x2,y1=y1,y2=y2)
        M.roi_index += 1
        self.Main.reload()
        M.refresh()
        self.Profile.flag_save = True
        #set_text(M.roi_entry,"")
    
def remove_roi():
    if messagebox.askokcancel("Remove roi", "Do you want to remove roi?"):
        if M.roi_index != -1:
            self.Profile.remove_roi(M.cam_pos,M.cmb_pos, M.roi_index)
            self.Main.reload()
            M.refresh()
            self.Profile.flag_save = True

def add_source():
    self.Profile.add_source(M.cam_pos)
    M.cam_pos = M.cam_pos + 1
    self.Main.reload()
    M.refresh()
    self.Profile.flag_save = True

def remove_source():
    self.Profile.remove_source(M.cam_pos)
    M.cmb_pos = M.cmb_pos - 1
    if M.cmb_pos < 0:
        M.cmb_pos = 0
        
    self.Main.reload()
    M.refresh()
    self.Profile.flag_save = True

def add_step():
    if messagebox.askokcancel("Add step", "Do you want to add step?"):
        self.Profile.add_step(M.cam_pos,M.cmb_pos)
        self.Main.set_step(M.cmb_pos + 1)

        self.Main.reload()
        M.refresh()
        self.Profile.flag_save = True


def add_step_above():
    if messagebox.askokcancel("Insert step", "Do you want to insert step?"):
        self.Profile.insert_step(M.cam_pos,M.cmb_pos)
        self.Main.set_step(M.cmb_pos)
        
        self.Main.reload()
        M.refresh()
        self.Profile.flag_save = True
    
def remove_step():
    if messagebox.askokcancel("Remove step", "Do you want to remove step?"):
        self.Profile.remove_step(M.cam_pos,M.cmb_pos)
        M.cmb_pos = M.cmb_pos - 1
        if M.cmb_pos < 0:
            M.cmb_pos = 0        
        self.Main.set_step(M.cmb_pos)

        self.Main.reload()
        M.refresh()
        self.Profile.flag_save = True

def duplicate_step():
    if messagebox.askokcancel("Duplicate Step", "Do you want duplicate step?"):
        new_pos = M.cmb.current()+1
        self.Profile.duplicate_step(M.cmb_cam.current(), M.cmb.current(), new_pos)      
        self.Main.set_step(new_pos)
        self.Main.reload()
        M.refresh()
        self.Profile.flag_save = True

def btn1_clicked(event):
    M.mv_type = ""
    
    if not self.Users.login_admin():
        return
        
    M.lx, M.ly = (M.view.viewer.canvasx(event.x), M.view.viewer.canvasy(event.y))
    
    ret, roi = self.Profile.get_selected_roi()
    if not ret:
        return
        
    M.lx1 = roi['x1'] 
    M.ly1 = roi['y1'] 
    M.lx2 = roi['x2'] 
    M.ly2 = roi['y2'] 
    
    if M.roi_index < 0:
        return
        
    scale = M.view.get_scale()
    roi = self.Profile.loaded_profile[M.cam_pos][M.cmb_pos]["roi"][M.roi_index]

    x1 = roi['x1'] * scale
    y1 = roi['y1'] * scale
    x2 = roi['x2'] * scale
    y2 = roi['y2'] * scale
    
    zp = 5 # resize activate zone pixel

    if x2+ zp>= M.lx > x2- zp and y2+ zp > M.ly > y2- zp and M.cf.move_all_var.get():
        M.mv_type = "resize_all"
        M.lrois = copy.deepcopy(self.Profile.loaded_profile[M.cam_pos][M.cmb_pos]["roi"])

    elif x2- zp >= M.lx >= x1 and y2- zp >= M.ly >= y1 and M.cf.move_all_var.get():
        M.mv_type = "move_all"
        M.lrois = copy.deepcopy(self.Profile.loaded_profile[M.cam_pos][M.cmb_pos]["roi"])
        
    elif x2- zp >= M.lx >= x1 and y2- zp >= M.ly >= y1:
        M.mv_type = "move"
        
    elif x2+ zp>= M.lx > x2 - zp and y2+ zp > M.ly > y2 - zp:
        M.mv_type = "resize"

    Logging.info(f'Mouse btn1 mv_type = {M.mv_type}')


def reset_roi_position():
    if not self.start.Users.login_admin():
        return
        
    ret, roi = self.start.Profile.get_selected_roi()
    if not ret:
        return 
        
    roi['x1'],roi['x2'],roi['y1'],roi['y2'] = self.start.MainUi.view.view_to_image_position(10,100,10,100)

    self.Main.reload()
    M.refresh()
    self.Profile.flag_save = True    
        
def btn1_move(event):
    scale = M.view.get_scale()

    x, y = (M.view.viewer.canvasx(event.x), M.view.viewer.canvasy(event.y))
    
    px = int(x/scale)
    py = int(y/scale)
    plx = int(M.lx/scale)
    ply = int(M.ly/scale)
    
    delta_x = px - plx
    delta_y = py - ply
    
    if M.roi_index < 0:
        return
        
    roi = self.Profile.loaded_profile[M.cam_pos][M.cmb_pos]["roi"][M.roi_index]
    
    if M.mv_type == "move":
        roi['x1'] = int(M.lx1 + px - plx) 
        roi['y1'] = int(M.ly1 + py - ply) 
        roi['x2'] = int(M.lx2 + px - plx) 
        roi['y2'] = int(M.ly2 + py - ply) 
        
        self.Main.reload()
        M.refresh()
        self.Profile.flag_save = True
        
    if M.mv_type == "resize":
        roi['x2'] = int(M.lx2 + px - plx)
        roi['y2'] = int(M.ly2 + py - ply)
        
        self.Main.reload()
        M.refresh()
        self.Profile.flag_save = True
        
    if M.mv_type == "move_all":       
        for roi_n, roi_aux in enumerate(self.Profile.loaded_profile[M.cam_pos][M.cmb_pos]["roi"]):
            roi_aux['x1'] = int(M.lrois[roi_n]['x1'] + delta_x) 
            roi_aux['y1'] = int(M.lrois[roi_n]['y1'] + delta_y) 
            roi_aux['x2'] = int(M.lrois[roi_n]['x2'] + delta_x) 
            roi_aux['y2'] = int(M.lrois[roi_n]['y2'] + delta_y) 
       
        self.Main.reload()
        M.refresh()
        self.Profile.flag_save = True
        
    if M.mv_type == "resize_all":
        for roi_n, roi_aux in enumerate(self.Profile.loaded_profile[M.cam_pos][M.cmb_pos]["roi"]):
            roi_aux['x1'] = int(M.lrois[roi_n]['x1'] * (delta_x*0.001) + M.lrois[roi_n]['x1']) 
            roi_aux['y1'] = int(M.lrois[roi_n]['y1'] * (delta_y*0.001) + M.lrois[roi_n]['y1']) 
            roi_aux['x2'] = int(M.lrois[roi_n]['x2'] * (delta_x*0.001) + M.lrois[roi_n]['x2']) 
            roi_aux['y2'] = int(M.lrois[roi_n]['y2'] * (delta_y*0.001) + M.lrois[roi_n]['y2'] )

        self.Main.reload()
        M.refresh()
        self.Profile.flag_save = True

    self.MainUi_Refresh.refresh_edit_roi_rectangle()
        
def rename_roi(): 
    if self.start.Users.login_admin():
        if M.roi_entry.get() == "":
            messagebox.showerror(title="Please enter roi name", message="Please enter roi name")
        else:   
            self.Profile.update_roi(M.cam_pos,M.cmb_pos, M.roi_index,{"name":f"{M.roi_entry.get()}"})

            self.Main.reload()
            M.refresh()
            self.Profile.flag_save = True
            
def step_move_up():
    if M.cmb.current() > 0:
        if messagebox.askokcancel("Move Step", "Do you want move step up?"):
            new_pos = M.cmb.current()-1
            self.Profile.move_step(M.cmb_cam.current(), M.cmb.current(), new_pos)      
            self.Main.set_step(new_pos)
            self.Main.reload()
            M.refresh()
            self.Profile.flag_save = True
            
def step_move_down():
    if M.cmb.current() < len(self.Profile.loaded_profile[M.cmb_cam.current()]):
        if messagebox.askokcancel("Move Step", "Do you want move step down?"):
            new_pos = M.cmb.current()+1
            self.Profile.move_step(M.cmb_cam.current(), M.cmb.current(), new_pos)             
            self.Main.set_step(new_pos)
            self.Main.reload()
            M.refresh()
            self.Profile.flag_save = True    

def roi_move_up():
    print(M.roi_index)
    if M.roi_index > 0:        
        self.Profile.move_roi(M.cmb_cam.current(), M.cmb.current(),M.roi_index , M.roi_index -1) 
        M.roi_index = M.roi_index -1
        self.Main.reload()
        M.refresh()
        self.Profile.flag_save = True
            
def roi_move_down():
    if M.roi_index > -1 and M.roi_index < len(self.Profile.loaded_profile[M.cmb_cam.current()][M.cmb.current()]["roi"]):
        self.Profile.move_roi(M.cmb_cam.current(), M.cmb.current(),M.roi_index , M.roi_index +1) 
        M.roi_index = M.roi_index +1
        self.Main.reload()
        M.refresh()
        self.Profile.flag_save = True
                
                    
M.view.viewer.bind("<B1-Motion>", btn1_move)   
M.view.viewer.bind("<Button-1>", btn1_clicked)   

     
M.btn_add_roi.configure(command=add_roi) 
M.btn_remove_roi.configure(command=remove_roi) 
M.btn_rename_roi.configure(command=rename_roi)

M.btn_add_step.configure(command=add_step) 
M.btn_add_step_above.configure(command=add_step_above)
M.btn_remove_step.configure(command=remove_step) 
self.MainUi.cf.btn_duplicate_step.configure(command=duplicate_step)

M.btn_add_source.configure(command=add_source) 
M.btn_remove_source.configure(command=remove_source) 

M.cf.btn_up_step.config(command = step_move_up)
M.cf.btn_down_step.config(command = step_move_down)

M.cf.btn_roi_up.config(command = roi_move_up)
M.cf.btn_roi_down.config(command = roi_move_down)