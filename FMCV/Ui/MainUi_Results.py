from FMCV.Ui import MainUi as M
from tkinter import messagebox
self = M.self
import sys


def btn1_clicked(event):
    x, y  = (M.r_view.viewer.canvasx(event.x), M.r_view.viewer.canvasy(event.y))
    
    selected_rect = -1
    nearest_dis = sys.maxsize
    
    for val in M.r_view.viewer.find_all():
        coords = M.r_view.viewer.coords(val) #val must be INT32
        if len(coords) ==4:
            yes, dis = in_region(x, y, coords)
            if yes and dis < nearest_dis:
                selected_rect = val
                nearest_dis = dis

    if selected_rect > -1:
        #result_roi = self.Main.results[M.cam_pos][M.cmb_pos][int(M.r_view.viewer.gettags(selected_rect)[0])]
        result_roi = self.Main.results[M.cam_pos][M.cmb_pos][int(M.r_view.viewer.gettags(selected_rect)[0])]#M.r_view.current_results[int(M.r_view.viewer.gettags(selected_rect)[0])]
        M.result_frame.result_roi = result_roi
        M.result_frame.update_results(result_roi)
        
def in_region(x,y,c):
    '''
        move click x,y
        coordinates of rectangle c x1, y1, x2, y2
        return Yes, distant
    '''
    #print("{} {}".format(xy,coords))
    if (x>=c[0] and x<=c[2] and
        y>=c[1] and y<=c[3]):
            distant = sys.maxsize
            if x - c[0] < distant:
                distant = x - c[0]
            if c[2] - x < distant:
                distant = c[2] - x
            if y - c[1] < distant:
                distant = y - c[1]
            if c[3] - y < distant:
                distant = c[3] - y
            return True, distant
    else:
        return False, -1
    
    
def btn1_move(event):    
    x,y = (M.r_view.viewer.canvasx(event.x), M.r_view.viewer.canvasy(event.y))
        

M.r_view.viewer.bind("<B1-Motion>", btn1_move)   
M.r_view.viewer.bind("<Button-1>", btn1_clicked) 