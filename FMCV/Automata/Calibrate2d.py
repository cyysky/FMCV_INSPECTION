import time
from FMCV import Logging
from FMCV.Cv import Match,Cv
from tkinter import simpledialog
from tkinter import messagebox
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np

import tkinter as tk
#f = plt.figure()
#f.set_figwidth(30)
#f.set_figheight(30)
import json

start = None

def init(in_start):
    global start
    start = in_start
    start.sub("tool/calibrate2d/calibrate", calibrate)
    start.sub("tool/calibrate2d/test", test_calibrated)
    start.sub("tool/calibrate2d/test_roi_center", test_roi_center)
    start.sub("tool/calibrate2d/set_user_coordinate", set_user_coordinate)
    
def calibrate():
    global start
    
    if not start.Users.login_admin():
        return 
        
    start.Com.select_user_coordinate_system(10) # FMCV Board selected
    start.Com.select_tool_coordinate_system(10) # FMCV Camera selected 
    
    messagebox.showinfo("XY Calibration", "Make sure FMCV Board and FMCV Camera selected for tool and user coordinate system")
    
    ori_pos = start.Com.get_tcp()
    print(ori_pos)
    
    ix, iy = get_center()
    print("Initial center ", ix,",", iy)

    if ix is None:
        Logging.error("No center detected")
        messagebox.showinfo("XY Calibration", "No center detected")
        return
        
    move_range_mm = int(simpledialog.askstring(title="XY Calibration", prompt="Moving Range :"))
    
    data_points = []
    for i in range(50):
        xr ,yr = move_random(ori_pos, move_range_mm)

        ixr, iyr = get_center()
        random_tcp = start.Com.get_tcp()
        print("Random ",xr ,", ",yr," Center ", ixr,",", iyr," @ Tcp ",random_tcp)
        
        if not [v for v in (ix, iy, ixr, iyr) if v is None]:
            data_point = {}
            o = ori_pos
            r = random_tcp
            data_point.update({"ix":ix, "iy":iy, "ix2":ixr-ix, "iy2":iyr-iy})
            data_point.update({"jx":o[0], "jy":o[1]})
            data_point.update({"jx2":xr, "jy2":yr})
            data_points.append(data_point)
        
        start.Com.to_tcp([xr*-1,yr*-1,0,0,0,0]) #move back
        user_feedback = messagebox.askyesno("XY Calibration", f"Move {i+1} sent to cobot, continue?")
        print(user_feedback)
        if not user_feedback:
            break
        
    for data in data_points:
        print(data)       
    
    #with open("data.txt", "w") as fp:
    #    json.dump(data_points, fp)  # encode dict into JSON
    
    if not (len(data_points)>5):
        messagebox.showinfo("XY Calibration", "Data must more then 6")
        return
    
    s, theta, reverse_x, dx, dy, error = calculate_rotation_xy(data_points)
    if error > 100:
        s, theta, reverse_x, dx, dy, error = calculate_rotation_xy(data_points,reverse_z = False)
    calibrated = {'s':s, 'theta':theta, 'reverse_x': reverse_x, 'dx':dx, 'dy':dy, 'o':[ori_pos[0],ori_pos[1]], 'i':[ix, iy]}
    print(f"s={s}, theta={theta}, dx={dx}, dy={dy} o={[ori_pos[0],ori_pos[1]]} i={[ix, iy]}")
    ret, roi = start.Profile.get_selected_roi()
    if ret:
        roi.update({"2d_calibrate":[s,theta,reverse_x,dx,dy]})
        roi.update({"2d_error_mm":error})
        start.ActionUi.reload_detection()
        start.ActionUi.save_profile()

        
def move_random(ori_pos, distant):
    p = ori_pos
    x1 = random.uniform(-distant, distant)
    y1 = random.uniform(-distant, distant)
    start.Com.to_tcp([x1,y1,0,0,0,0])#,speed=1000,accel=100)
    return x1, y1
    
def wait_reach_position():
    time.sleep(0.2)
    # timeout variable can be omitted, if you use specific value in the while condition
    timeout = 10   # [seconds]
    timeout_start = time.time()
    while time.time() < timeout_start + timeout:
        if start.Com.get_status('reach'):
            break
    time.sleep(0.05)

def get_center():
    global start
    ret, roi = start.Profile.get_selected_roi()
    if ret:
        tmp = Cv.to_gray(roi["img"])
        frame = list(start.Cam.get_image().values())[start.MainUi.cam_pos]
        
        if roi.get('K') and roi.get('D'):
            mtx = np.array(roi.get('K'))
            dist = np.array(roi.get('D'))
            h,  w = frame.shape[:2]
            newcameramtx, img_roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
            dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
            # crop the image
            ax, ay, aw, ah = img_roi
            dst = dst[ay:ay+ah, ax:ax+aw]
            frame = cv2.resize(dst, (w,h), interpolation = cv2.INTER_AREA)

        frame_gray = Cv.to_gray(frame)
        print(frame_gray.shape)
        print(roi["rotate"])
        frame_gray = Cv.get_rotate(roi["rotate"],frame_gray)
        if roi["blur"] > 0:
            frame_gray = cv2.blur(frame_gray ,(roi["blur"], roi["blur"]))
            tmp = cv2.blur(tmp ,(roi["blur"], roi["blur"]))
            
        if np.all([roi['mask'] == 255]):
            mask = None
            print("White template")
        else:
            mask = roi['mask']
            print("Masked template") 
            
        start_time = time.time()
        res,top_left,bottom_right,top_val,cropped = start.Cv.match_template(roi['img'] , frame_gray, Mask=mask, BlurSize=roi['blur'], Mode=roi['method'])
        
        #points_list = Match.modifiedMatchTemplate(frame_gray,tmp, "TM_CCOEFF_NORMED",  0, None, [0,1], 1, [100,101], 1, True, True)
        # if np.all([roi['mask'] == 255]):
            # mask = None
            # print("White template")
        # else:
            # mask = roi['mask']
        # print("Masked template") 
        # res,top_left,bottom_right,top_val,cropped = start.Cv.match_template(roi['img'] , frame_gray , Mask=mask,BlurSize=roi['blur'],Mode=roi['method'])
        end_time = time.time()
        print(f"time match consumed {end_time - start_time}")
        center_x = int((top_left[0] + bottom_right[0])/2)
        center_y = int((top_left[1] + bottom_right[1])/2)
        print(center_x,center_y)

        start.MainUi.update_result_frame(cv2.circle(frame, (center_x,center_y), radius=0, color=(0, 0, 255), thickness=30))
        
        
        return center_x, center_y
        # centers_list = []
        # for point_info in points_list[:1]:
            # point = point_info[0]
            # angle = point_info[1]
            # scale = point_info[2]
            # width = point_info[4]
            # height = point_info[5]
            # print(point_info[3])
            # centers_list.append([point, scale])
            # center_x = int(point[0] + (width/2)*scale/100)
            # center_y = int(point[1] + (height/2)*scale/100)
            # print(center_x,center_y)
            # #cv2.imshow("h",cv2.circle(frame, (center_x,center_y), radius=0, color=(0, 0, 255), thickness=30))
            # #cv2.waitKey(0)
            # start.MainUi.r_view.set_image(cv2.circle(frame, (center_x,center_y), radius=0, color=(0, 0, 255), thickness=30))
            # start.MainUi.top.update()
            # return center_x, center_y
    return None, None
        
        
def calculate_rotation_xy(data_points, reverse_z = True):
    
    if reverse_z:
        reverse_x = -1
    else:
        reverse_x = 1

    ix = np.array([x['ix2'] for x in data_points])
    iy = np.array([reverse_x * x['iy2'] for x in data_points]) # minus because camera z axis and robot z axis are opposite, so it's not a real 2d problem
   
    jx = np.array([x['jx2'] for x in data_points])
    jy = np.array([x['jy2'] for x in data_points])

    '''
    problem: 
            image XY to robot XY transformation seems to be a 2d rotation + translation + scaling 
            jx = s * (cosθ * ix - sinθ * iy) + dx
            jy = s * (sinθ * ix + cosθ * iy) + dy
            given ix,iy,jx and jy, compute s, θ, dx, dy
    solution:
            since data is noisy, use nonlinear optimization(Gauss Newton Method) to compute the optimal solution
    '''

    H = np.eye(4)
    b = np.zeros((4,1))
    #initial values
    s, theta, dx, dy = 1, 0, 0, 0
    max_iter = 60

    def evaluate_error(ix,iy,jx,jy,s,theta,dx,dy):
        b = np.zeros((4,1),dtype=np.float64)
        H = np.zeros((4,4),dtype=np.float64)
        cost = 0
        for i in range(len(ix)):
            ixi,iyi,jxi,jyi = ix[i],iy[i],jx[i],jy[i]
            residual = np.array([[s * (np.cos(theta) * ixi - np.sin(theta) * iyi) + dx - jxi],
                                 [s * (np.sin(theta) * ixi + np.cos(theta) * iyi) + dy - jyi]])
            jacobian = np.array([[np.cos(theta) * ixi - np.sin(theta) * iyi, s * (-ixi * -np.sin(theta) - iyi*np.cos(theta)),1,0],
                                 [np.sin(theta) * ixi + np.cos(theta) * iyi, s * (ixi * np.cos(theta) - iyi*np.sin(theta)),0,1]])
            cost += np.linalg.norm(residual)**2
            H += jacobian.transpose()@jacobian
            b += jacobian.transpose()@residual

        return cost,H,b


    error,H,b = evaluate_error(ix,iy,jx,jy,s,theta,dx,dy)
    print('initial:',error,H,b)
    for iter in range(max_iter):
        step = -np.linalg.solve(H,b)
        if np.linalg.norm(step)<1E-6:
            break

        s += step[0][0]
        theta += step[1][0]
        dx += step[2][0]
        dy += step[3][0]
        error, H, b = evaluate_error(ix, iy, jx, jy, s, theta, dx, dy)
        print('iter:',iter,' error:',error)

    print(s,theta*360/np.pi,dx,dy)
    print(len(data_points))

    fig, ax = plt.subplots()

    fig.set_figwidth(30)
    fig.set_figheight(30)

    transformed_ix,transfoemed_iy = [],[]
    for i in range(len(ix)):
        transformed_ix.append(s * (np.cos(theta) * ix[i] - np.sin(theta) * iy[i]) + dx)
        transfoemed_iy.append(s * (np.sin(theta) * ix[i] + np.cos(theta) * iy[i]) + dy)
        print(ix[i],iy[i],jx[i],transformed_ix[i],jy[i],transfoemed_iy[i])
    #ax.plot(ix, iy,'r')
    ax.plot(np.array(transformed_ix), np.array(transfoemed_iy),'b')
    ax.plot(jx, jy,'g')


    # for i in range(len(data_points)):
    #     ax.text(data_points[i]['ix'],-data_points[i]['iy'],str(i))

    plt.show()
    
    return s, theta, reverse_x,  dx, dy, error
 
def get_offset_mm(off_x, off_y, calibrated_list):
    Logging.debug("off_x, off_y, calibrated_list", off_x, off_y, calibrated_list)
    ix = off_x
    iy = off_y
    c = calibrated_list
    calibrated={'s':c[0], 'theta':c[1],'reverse_x':c[2], 'dx':c[3], 'dy':c[4]}
    jx = calibrated['s'] * (np.cos(calibrated['theta']) * ix - np.sin(calibrated['theta']) * -iy )+ calibrated['dx']
    jy = calibrated['s'] * (np.sin(calibrated['theta']) * ix + np.cos(calibrated['theta']) * -iy) + calibrated['dy']
    if calibrated['reverse_x'] == 1:
        return -jx, jy
    else:
        return -jx, -jy


def set_user_coordinate():
    ret, roi = start.Profile.get_selected_roi()
    if not ret:
        return 
        
    if roi.get('R') is None:
        messagebox.showwarning("Warning", "Need 3D calibrate first")
        return

    if not messagebox.askyesno("Position Confirmation", "Are you sure you want to set current position as FMCV 2D user coordinate?"):
        return

    start.Com.select_user_coordinate_system(0) # World Frame selected
    start.Com.select_tool_coordinate_system(0) # End Flage selected 
    messagebox.showwarning("Warning", "Please select World Frame and End Flage\nand Please close JAKA Zu software before proceed\n")
    current_pos = start.Com.get_tcp()
    eye_to_hand_R = np.array(roi.get('R'))
    eye_to_hand_t = np.array(roi.get('T'))
    
    user_coordinate_offset = start.Calibrate3d.getCameraPoseGivenHandPose(current_pos, eye_to_hand_R, eye_to_hand_t)

    tool_coordinate_offset = start.Calibrate3d.getHandToCameraCoordinate(eye_to_hand_R, eye_to_hand_t)
    
    if (
        start.Com.set_tool_coordinate_system(tool_coordinate_offset,10,"FMCV Camera") and
        start.Com.set_user_coordinate_system(user_coordinate_offset,9,"FMCV 2D")
        ):
        messagebox.showinfo("Set user and tool coordinate","Done update FMCV 2D UCS and FMCV Camera TCP successfully, please verify is all 0.000mm")

def test_roi_center():

    ret, roi = start.Profile.get_selected_roi()
    if not ret:
        return 
        
    if roi.get('2d_calibrate') is None:
        messagebox.showwarning("Warning", "ROI not 2d calibrated")
        return 
        
    if not start.Users.login_user():
        return
    messagebox.showwarning("Warning", "Make sure FMCV Board and FMCV Camera selected for tool and user coordinate system")

        
    frm =  list(start.Cam.get_image().values())[start.MainUi.cam_pos]
    
    frm = Cv.get_rotate(roi['rotate'],frm)
    
    h,w = frm.shape[:2]
    
    if roi.get('K') and roi.get('D'):
        mtx = np.array(roi.get('K'))
        dist = np.array(roi.get('D'))
        newcameramtx, img_roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        dst = cv2.undistort(frm, mtx, dist, None, newcameramtx)
        ax, ay, aw, ah = img_roi
        dst = dst[ay:ay+ah, ax:ax+aw]
        frm = cv2.resize(dst, (w,h), interpolation = cv2.INTER_AREA)
        
    m = roi['margin']
    
    x1 = roi['x1'] - m
    y1 = roi['y1'] - m
    x2 = roi['x2'] + m
    y2 = roi['y2'] + m
    
    print(f"x1 {x1} y1 {y1} x2 {x2} y2 {y2}")
    
    xm = m
    ym = m
    
    if x1 < 0 : xm = m + x1
    if y1 < 0 : ym = m + y1
    
    print(f"xm {xm},{ym}")
    
    if x1 < 0 : x1 = 0 
    if y1 < 0 : y1 = 0 
    if x2 > w : x2 = w 
    if y2 > h : y2 = h
    

    if np.all([roi['mask'] == 255]):
        mask = None
        print("White template")
    else:
        mask = roi['mask']
        print("Masked template") 

    res,top_left,bottom_right,top_val,cropped = start.Cv.match_template(roi['img'] , frm[y1:y2,x1:x2] , Mask=mask, BlurSize=roi['blur'], Mode=roi['method'])
    
    center_x = int((roi['x1']+roi['x2'])/2)+top_left[0] - xm
    center_y = int((roi['y1']+roi['y2'])/2)+top_left[1] - ym
    
    start.MainUi.update_result_frame(cv2.circle(frm.copy(), (center_x,center_y), radius=0, color=(0, 0, 255), thickness=30))

    angle = -999
    if roi.get("angle_enable"):
        a = Match.match_rotate(roi.get('img'),frm[y1:y2,x1:x2].copy())
        if a:
            frame, cropped_r, [xm_center, ym_center], angle , dst, xy_dst = a
            frame = cv2.circle(frame, (center_x - x1 ,center_y - y1), radius=0, color=(0, 0, 255), thickness=20)
            start.MainUi.update_result_frame(frame)
    
    start.Com.select_user_coordinate_system(10) # FMCV Chessboard selected
    start.Com.select_tool_coordinate_system(10) # FMCV Camera selected 
    
    if angle > -999:
        cx = int(frm.shape[1]/2)
        cy = int(frm.shape[0]/2)                    
        Logging.info(f"cx {cx},{cy}")
        
        cx_roi = int( x1 + xm + int((roi['x2']-roi['x1'])/2))
        cy_roi = int( y1 + ym + int((roi['y2']-roi['y1'])/2))
        Logging.info(f"cx_roi {cx_roi},{cy_roi}")
        
        print("x1 xm ym",x1,xm,ym)
        x_center = x1 + xm_center
        y_center = y1 + ym_center
        Logging.info(f"x_center {x_center},{y_center}")
        
        theta = -angle * np.pi / 180
        # cx_roi_rot = cx + (cx_roi - cx)*np.cos(theta) - (cy_roi - cy)*np.sin(theta)
        # cy_roi_rot = cy + (cx_roi - cx)*np.sin(theta) + (cy_roi - cy)*np.cos(theta)
        # Logging.info(f"cx_roi_rot {cx_roi_rot},{cy_roi_rot}")

        x_rot = cx + (x_center - cx)*np.cos(theta) - (y_center - cy)*np.sin(theta)
        y_rot = cy + (x_center - cx)*np.sin(theta) + (y_center - cy)*np.cos(theta)
        Logging.info(f"x_rot {x_rot},{y_rot}")
        
        x_offset = x_rot - cx_roi
        y_offset = y_rot - cy_roi
        Logging.info(f"x_offset {x_offset},{y_offset}")
        
        off_x_mm, off_y_mm = get_offset_mm(x_offset, y_offset, roi['2d_calibrate'])
        start.Com.to_tcp([off_x_mm, off_y_mm, 0, 0, 0, angle])
        Logging.info([off_x_mm, off_y_mm, 0, 0, 0, angle])
    else:
        off_x_mm, off_y_mm = get_offset_mm(top_left[0] - xm, top_left[1] - ym, roi['2d_calibrate'])
        start.Com.to_tcp([off_x_mm, off_y_mm, 0, 0, 0, 0])
        Logging.info([off_x_mm, off_y_mm, 0, 0, 0, 0])
    messagebox.showinfo("Move to ROI Center", "Move to ROI Center sent to Cobot")
 
def test_calibrated():
    ret, roi = start.Profile.get_selected_roi()
    if not ret:
        return
        
    if roi.get('2d_calibrate') is None:
        messagebox.showwarning("Warning", "ROI not calibrated")
        return 
        
    if not start.Users.login_user():
        return

    messagebox.showwarning("Warning", "Make sure FMCV Board and FMCV Camera selected for tool and user coordinate system")
    
    frame = list(start.Cam.get_image().values())[start.MainUi.cam_pos]
    Logging.info(frame.shape)
    frame = Cv.get_rotate(roi['rotate'],frame)

        
    ix, iy = get_center()
    angle = -999
    if roi.get("angle_enable"):
        a = Match.match_rotate(roi.get('img'),frame.copy())
        if a:
            frame, cropped_r, [x_center, y_center], angle , dst, xy_dst = a
            

    
    start.Com.select_user_coordinate_system(10) # FMCV Board selected
    start.Com.select_tool_coordinate_system(10) # FMCV Camera selected 
    
    if angle > -999:
        cx = int(frame.shape[1]/2)
        cy = int(frame.shape[0]/2)                    
        Logging.info(f"cx {cx},{cy}")

        Logging.info(f"x_center {x_center},{y_center}")

        x_offset = x_center - cx
        y_offset = y_center - cy
        Logging.info(f"x_offset {x_offset},{y_offset}")
        
        off_x_mm, off_y_mm = get_offset_mm(x_offset, y_offset, roi['2d_calibrate'])
        start.Com.to_tcp([off_x_mm, off_y_mm, 0, 0, 0, angle])
        Logging.info([off_x_mm, off_y_mm, 0, 0, 0, angle])
    else:
        offset_x = ix-(int(frame.shape[1]/2)) #960 is half of 1920, 712 is train center back to x center
        offset_y = iy-(int(frame.shape[0]/2))#-(292-600) #600 is half of 1200, 644 is train center back to y center'

        off_x_mm, off_y_mm = get_offset_mm(offset_x, offset_y, roi['2d_calibrate'])
        
        start.Com.to_tcp([off_x_mm, off_y_mm, 0, 0, 0, 0])
        Logging.info([off_x_mm, off_y_mm, 0, 0, 0, 0])
        
        frame = cv2.circle(frame, (ix,iy), radius=0, color=(0, 0, 255), thickness=20)
        
    start.MainUi.update_result_frame(frame)
    messagebox.showinfo("Move to Global Center", "Move to Global Center sent to Cobot")