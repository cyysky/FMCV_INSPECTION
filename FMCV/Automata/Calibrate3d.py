import time
from FMCV import Logging
from FMCV.Cv import Match,Cv
from tkinter import simpledialog
from tkinter import messagebox

import random
import numpy as np
import cv2

import json
from scipy.spatial.transform import Rotation

import glob
import math
import sys
try:
    import open3d as o3d
except ImportError:
    import pyvista as pv
    

from pathlib import Path
import os

calibrated = None
start = None

# ROW = 20
# COL = 20
# OFFSET_Y = 0
# ROW = 8
# COL = 5
#OFFSET_Y = -10

def init(in_start):
    global start
    start = in_start
    start.sub("tool/calibrate3d/move_to_chessboard", move_to_chessboard)
    start.sub("tool/calibrate3d/calculate", calculate)
    start.sub("tool/calibrate3d/calibrate", calibrate)
    start.sub("tool/calibrate3d/get_data", take_data)
    start.sub("tool/calibrate3d/set_user_coordinate", set_user_coordinate)
    start.sub("tool/calibrate3d/get_user_frame", get_user_frame)
    start.sub("tool/calibrate3d/reset_user_coordinate", reset_user_coordinate)

def get_tcp():
    tcp = start.Com.get_tcp()
    if tcp is None:
        messagebox.showwarning("Warning","Cobot disconnected, check connection")
    return tcp

def set_user_coordinate():
    ret, roi = start.Profile.get_selected_roi()
    if ret:
        if roi.get('R') is not None:
            messagebox.showwarning("Warning", "Please select World Frame and End Flage\nand Verify FMCV Chessboard user coordinate is all 0.000mm\nand Close JAKA Zu software before proceed")            
            move_to_chessboard()
            current_pos = get_tcp()
            eye_to_hand_R = np.array(roi.get('R'))
            eye_to_hand_t = np.array(roi.get('T'))
            
            user_coordinate_offset = getCameraPoseGivenHandPose(current_pos, eye_to_hand_R, eye_to_hand_t)

            tool_coordinate_offset = getHandToCameraCoordinate(eye_to_hand_R, eye_to_hand_t)
            
            if (
                start.Com.set_tool_coordinate_system(tool_coordinate_offset,10,"FMCV Camera") and
                start.Com.set_user_coordinate_system(user_coordinate_offset,10,"FMCV Board")
                ):
                messagebox.showinfo("Set user and tool coordinate","Done update FMCV Chessboard UCS and FMCV Camera TCP successfully, please verify is all 0.000mm")
                
        else:
            messagebox.showwarning("Warning", "Need calibrate first")

def reset_user_coordinate():
    messagebox.showwarning("Warning", "Please close JAKA Zu software before proceed") 
    if (start.Com.set_tool_coordinate_system([0,0,0,0,0,0],10,"FMCV Camera") and
        start.Com.set_user_coordinate_system([0,0,0,0,0,0],10,"FMCV Board")
        ):
        messagebox.showinfo("Reset user coordinate","Successfully reset tool and user coordinate system")
                       
def move_to_chessboard():
    ret, roi = start.Profile.get_selected_roi()
    if ret:
        if roi.get('R') is not None:
            start.Com.select_user_coordinate_system(0) # World Frame selected
            start.Com.select_tool_coordinate_system(0) # End Flage selected 
            messagebox.showinfo("Please verify","Now start moving make sure World Frame and End Flage selected for tool and user coordinate system")
            time.sleep(0.1)
            start.Com.to_tcp([0,0,0,0,0,0])
            tcp = get_tcp()
            if tcp is not None:
                output_pos, img_chessboard, pixel_to_mm = calculate(roi.get('R'), roi.get('T'), roi.get('K'), roi.get('D'),
                                                        tcp, offset_r_xyz = roi.get('offset_r_xyz'),
                                                        box_rows_cols = roi.get('box_rows_cols'), box_size = roi.get('box_size'))
                                                        
                if output_pos:
                    roi['pixel_to_mm']=pixel_to_mm
                    start.MainUi.update_result_frame(img_chessboard)
                    start.Com.to_tcp(output_pos, speed=30, accel=100, relative_move=False)
                    messagebox.showinfo("Move to chessboard","Move to chessboard command sent to Cobot")
                else:
                    messagebox.showwarning("Warning", "No Chessboard Found")
        else:
            messagebox.showwarning("Warning", "Need calibrate first")
            
def get_user_frame():
    ret, roi = start.Profile.get_selected_roi()
    if ret:
        if roi.get('R') is not None:
            world_frame = getUserCoordinateParameters(0,-10,roi.get('offset_z'),roi.get('W'),roi.get('Q'))
            start.Com.send_tcp_modbus(world_frame)
            print(world_frame)
            
#usage:
#input: end effector pose: rxe,rye,rze,jxe,jye,jze
#output: camera pose: rxc,ryc,rzc,jxc,jyc,jzc
def getCameraPoseGivenHandPose( tcp ,R_eye_to_hand,t_eye_to_hand):
    jxe,jye,jze,rxe,rye,rze = tcp
    R_hand_to_base = Rotation.from_euler('xyz', [rxe, rye, rze], degrees=True).as_matrix()
    t_hand_to_base = np.array([[jxe], [jye], [jze]])

    R_eye_to_base = R_hand_to_base@R_eye_to_hand
    t_eye_to_hand = R_hand_to_base@t_eye_to_hand + t_hand_to_base

    rxc,ryc,rzc = Rotation.from_matrix(R_eye_to_base).as_euler('xyz', degrees=True)
    jxc,jyc,jzc = t_eye_to_hand.tolist()
    return [jxc[0],jyc[0],jzc[0],rxc,ryc,rzc]

def getUserCoordinateParameters(shift_x, shift_y, offset_z, R_world_to_base, t_world_to_base):
    user_to_world_R = np.eye(3)
    user_to_world_t = np.array([shift_x,shift_y, offset_z]).reshape((3,1))

    user_to_base_R = R_world_to_base@user_to_world_R
    user_to_base_t = R_world_to_base@user_to_world_t + t_world_to_base

    rx,ry,rz = Rotation.from_matrix(user_to_base_R).as_euler('xyz',degrees=True)
    x,y,z = user_to_base_t[:,0]
    return [x,y,z,rx,ry,rz]

def getHandToCameraCoordinate(eye_to_hand_R, eye_to_hand_t):
    #hand_to_eye_R = eye_to_hand_R.transpose()
    #hand_to_eye_t = -hand_to_eye_R@eye_to_hand_t
    #rx,ry,rz = Rotation.from_matrix(hand_to_eye_R).as_euler('xyz', degrees=True)
    #jx,jy,jz = hand_to_eye_t.tolist()
    #tool_coordinate_offset = [jx[0],jy[0],jz[0],rx,ry,rz]
    rx,ry,rz = Rotation.from_matrix(eye_to_hand_R).as_euler('xyz', degrees=True)
    jx,jy,jz = eye_to_hand_t.tolist()
    tool_coordinate_offset = [jx[0],jy[0],jz[0],rx,ry,rz]
    return tool_coordinate_offset


def average_distance_between_points(points, pattern_size):
    total_distance = 0
    num_distances = 0

    #pattern_size = np.flip(pattern_size)
    # Calculate the average distance for horizontal adjacent points
    for row in range(pattern_size[0]):
        for col in range(pattern_size[1]-1):
            idx_a = (row * pattern_size[1])  + col 
            point_a = points[idx_a]
            idx_b = (row * pattern_size[1])  + col + 1
            point_b = points[idx_b]
            total_distance += np.linalg.norm(point_a - point_b)
            print()
            print(total_distance,idx_a,idx_b, point_a, point_b, np.linalg.norm(point_a - point_b))
            num_distances += 1

    # #Calculate the average distance for vertical adjacent points
    # for row in range(pattern_size[0] - 1):
       # for col in range(pattern_size[1]):
           # total_distance += np.linalg.norm(points[row * pattern_size[0] + col] - points[(row + 1) * pattern_size[0] + col])
           # print(np.linalg.norm(points[row * pattern_size[0] + col] - points[row * pattern_size[0] + col + 1]))
           # num_distances += 1

    return total_distance / num_distances


def calculate(R, T, K, D, tcp, box_rows_cols = [8,5],  
                offset_r_xyz = [0,0,150,0,0,0],
                box_size=30, raw_frame = None, points = None):
                
    if R is None:
        Logging.error('3D not calibrated')
        return None,None,None

    ROW = box_rows_cols[0]
    COL = box_rows_cols[1]
    

    camera_matrix = np.array(K)

    distortion_coefficients = np.array(D)

    #ROW = 8
    #COL = 5

    size = box_size
    print(f"Box Size {box_size}")

    offset_x = offset_r_xyz[0]
    offset_y = offset_r_xyz[1]
    offset_z = offset_r_xyz[2] 
    offset_rx = offset_r_xyz[3]
    offset_ry = offset_r_xyz[4]
    offset_rz = offset_r_xyz[5] 

    #desired_eye_to_world_R = np.eye(3)
    desired_eye_to_world_R = Rotation.from_euler('xyz',[offset_rx, offset_ry, offset_rz], degrees=True).as_matrix()
    #Hi Chong, if you want offset in angles as well, simply change this line 'desired_eye_to_world_R = np.eye(3)' to 
    #'desired_eye_to_world_R = Rotation.from_euler([rx,ry,rz],'xyz').as_matrix()'
    desired_height = offset_z
    #desired_eye_to_world_t = np.array([[(COL*0.5+1)*size+offset_x],[(ROW*0.5+1)*size+offset_y],[-desired_height]])
    #if points is not None:
    desired_eye_to_world_t = np.array([[(COL*0.5)*size+offset_x],[(ROW*0.5)*size+offset_y],[-desired_height]])
        
    tcp_data = tcp
    end_effector_pose = {}
    end_effector_pose.update({"jx":tcp_data[0], "jy":tcp_data[1], "jz":tcp_data[2],
                        "rx":tcp_data[3], "ry":tcp_data[4], "rz":tcp_data[5]})

    if raw_frame is None: 
        raw_frame = list(start.Cam.get_image().values())[start.MainUi.cam_pos]
    img = raw_frame

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ROW * COL, 3), np.float32)
    objp[:, :2] = np.mgrid[0:COL, 0:ROW].T.reshape(-1, 2) * size

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    pnp_flag = cv2.SOLVEPNP_ITERATIVE 

    if points is not None:
        pnp_flag = cv2.SOLVEPNP_IPPE_SQUARE 
        corners2 = points
        img_chessboard = img
        squareLength = box_size
        print("objp",objp)
        objp = np.array([[-squareLength / 2, squareLength / 2, 0], 
                 [ squareLength / 2, squareLength / 2, 0], 
                 [ squareLength / 2, -squareLength / 2, 0], 
                [-squareLength / 2, -squareLength / 2, 0]], dtype=np.float64)
        print("objp calculated",objp)
        print("corner_shape",corners2)
        #print(corners2)
        # Create empty mtx and dist
        #K = np.zeros((3, 3), dtype=np.float64)
        #D = np.zeros((1, 5), dtype=np.float64)
    else:
        # Find teye_to_hand ceye_to_handss board corners
        ret, corners = cv2.findChessboardCorners(gray, (COL, ROW), None)
        # If found, add object points, image points (after refining teye_to_handm)
        if not ret:
            Logging.error('no chessboard found')
            return None,None,None
            
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        print(corners2.shape)
        #print(corners2)
        
        img_chessboard = cv2.drawChessboardCorners(img, (COL, ROW), corners2, ret)
    
    # Calculate the average distance between adjacent corners in pixels
    avg_corner_distance_pixels = average_distance_between_points(corners2, box_rows_cols)
    print("avg_corner_distance_pixels:", avg_corner_distance_pixels)
    print(np.linalg.norm(corners2[0] - corners2[1]))
    # Calculate the pixel-to-millimeter conversion
    pixel_to_mm = box_size / avg_corner_distance_pixels

    print("Pixel to millimeter conversion:", pixel_to_mm)
    
    ret,world_to_eye_r,world_to_eye_t = cv2.solvePnP(objp,corners2,camera_matrix,distortion_coefficients, flags=pnp_flag)
    world_to_eye_R,_ = cv2.Rodrigues(world_to_eye_r)


    eye_to_hand_R = np.array(R)
    eye_to_hand_t = np.array(T)
    
    hand_to_eye_R = eye_to_hand_R.transpose()
    hand_to_eye_t = -hand_to_eye_R@eye_to_hand_t


    hand_to_base_R = Rotation.from_euler('xyz',[end_effector_pose['rx'],
                                                end_effector_pose['ry'],
                                                end_effector_pose['rz']],degrees = True).as_matrix()
    hand_to_base_t = np.array([[end_effector_pose['jx']],
                               [end_effector_pose['jy']],
                               [end_effector_pose['jz']]])


    def composeTransformation(R1,t1,R2,t2):
        assert R1.shape == (3, 3)
        assert R2.shape == (3, 3)
        assert t1.shape == (3, 1)
        assert t2.shape == (3, 1)
        R = R1@R2
        t = R1@t2+t1
        return R,t

    world_to_hand_R, world_to_hand_t = composeTransformation(eye_to_hand_R,eye_to_hand_t,world_to_eye_R,world_to_eye_t)
    world_to_base_R, world_to_base_t = composeTransformation(hand_to_base_R,hand_to_base_t,world_to_hand_R,world_to_hand_t)

    desired_eye_to_base_R,desired_eye_to_base_t = composeTransformation(world_to_base_R,world_to_base_t,desired_eye_to_world_R,desired_eye_to_world_t)
    desired_hand_to_base_R,desired_hand_to_base_t = composeTransformation(desired_eye_to_base_R,desired_eye_to_base_t,hand_to_eye_R,hand_to_eye_t)

    rx,ry,rz = Rotation.from_matrix(desired_hand_to_base_R).as_euler('xyz', degrees=True)
    jx,jy,jz = desired_hand_to_base_t

    Logging.info(jx[0],jy[0],jz[0],rx,ry,rz)
    return [jx[0],jy[0],jz[0],rx,ry,rz], img_chessboard ,pixel_to_mm

   
def calibrate():

    ret, roi = start.Profile.get_selected_roi()
    if ret:
    
        roi.pop("R", None)
        roi.pop("T", None)
        roi.pop("K", None)
        roi.pop("D", None)
        
        calibration_dataset_path = start.Profile.get_profile_folder_path() / 'calibration_data' / roi['name']
        
        # hand: robot arm end effector
        # eye: camera
        # base: robot arm base
        # world: chessboard coordinate
        # R: Rotation
        # t: position
        
        # ROW = 8
        # COL = 5
        box_rows_cols = roi.get('box_rows_cols')
        ROW = box_rows_cols[0]
        COL = box_rows_cols[1]
        
        size = float(roi['box_size'])

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((ROW * COL, 3), np.float32)
        objp[:, :2] = np.mgrid[0:COL, 0:ROW].T.reshape(-1, 2) * size
        # Arrays to store object points and image points from all teye_to_hand images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        images = sorted(calibration_dataset_path.glob('*.png'))
        images_loaded = []
        for fname in images:
            Logging.info(fname)
            iname = str(fname)
            img = cv2.imread(iname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find teye_to_hand ceye_to_handss board corners
            ret, corners = cv2.findChessboardCorners(gray, (COL, ROW), None)
            # If found, add object points, image points (after refining teye_to_handm)
            if ret == True:
                #print(corners)
                objpoints.append(objp)
                #corners2 = corners
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners)
                images_loaded.append(fname)
                # Draw and display teye_to_hand corners

                #cv2.drawCeye_to_handssboardCorners(img, (COL, ROW), corners2, ret)
                #cv2.imshow(fname, img)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
        # ret, mtx, dist, rvecs, tvecs = cv2.fiseye_to_handye.calibrate(objpoints, imgpoints, gray.shape[::-1], None, None)
        print(images_loaded)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        Logging.info(mtx)
        Logging.info(dist) 
        Logging.info(rvecs,len(rvecs))
        Logging.info(tvecs)
        
        json_file = calibration_dataset_path / 'calibration_dataset.json'

        eyes_to_world = {}
        for i in range(len(images_loaded)):
            filename = images_loaded[i].name #split('\\')[-1]
            world_to_eye_r = rvecs[i]
            world_to_eye_t = tvecs[i]
            world_to_eye_R,_ = cv2.Rodrigues(world_to_eye_r)
            eye_to_world_R = world_to_eye_R.transpose()
            eye_to_world_t = -eye_to_world_R@world_to_eye_t
            eyes_to_world[filename] = [eye_to_world_R,eye_to_world_t]

        hands_to_base = {}
        with open(json_file,'r') as file:
            data = json.load(file)
            for d in data['data']:
                if eyes_to_world.get(d["image"]) is not None:#previous : but some missed some NG datasets.
                    R_hand_to_base = Rotation.from_euler('xyz',[d['rx'],d['ry'],d['rz']],degrees = True).as_matrix()
                    t_hand_to_base = np.array([[d['jx']],[d['jy']],[d['jz']]])
                    hands_to_base[d["image"]] = [R_hand_to_base,t_hand_to_base]

        def skworld_to_eye_symmetric_matrix(v):
            assert v.shape == (3,1)
            return np.array([[0,-v[2][0],v[1][0]],
                             [v[2][0],0,-v[0][0]],
                             [-v[1][0],v[0][0],0]])

        def exponentialMap(v):
            R,_ = cv2.Rodrigues(v)
            return R

        #R_world_to_base = np.eye(3,dtype=np.float64)
        t_world_to_base = np.array([0.5,0.5,0.5]).reshape((3,1))
        R_eye_to_hand = np.eye(3,dtype=np.float64)
        t_eye_to_hand = np.array([0.5,0.5,0.5]).reshape((3,1))
        #scale = 1.0

        last_translation_rmse = 1e9


        def visualize(hands_to_base: dict, eyes_to_world: dict, R_eye_to_hand, t_eye_to_hand, t_world_to_base):
            points1 = []
            points2 = []
            for id in hands_to_base.keys():
                R_eye_to_world, t_eye_to_world = eyes_to_world[id]
                t_world_to_eye = -R_eye_to_world.transpose() @ t_eye_to_world

                R_hand_to_base, t_hand_to_base = hands_to_base[id]
                R_base_to_hand, t_base_to_hand = R_hand_to_base.transpose(), -R_hand_to_base.transpose() @ t_hand_to_base

                t_world_to_hand1 = R_eye_to_hand @ t_world_to_eye + t_eye_to_hand
                t_world_to_hand2 = R_base_to_hand @ t_world_to_base + t_base_to_hand

                points1.append(t_world_to_hand1)
                points2.append(t_world_to_hand2)

            if 'pv' in globals():
                # Create the point clouds
                cloud1 = pv.PolyData(points1)
                cloud2 = pv.PolyData(points2)

                # Create a plotter object and add the point clouds to it
                plotter = pv.Plotter()
                plotter.add_mesh(cloud1, color="red")
                plotter.add_mesh(cloud2, color="green")

                # Show the plot
                plotter.show()
            
            if 'o3d' in globals():
                pcd1 = o3d.geometry.PointCloud()
                pcd1.points = o3d.utility.Vector3dVector(points1)
                pcd1.paint_uniform_color([1, 0, 0])
                pcd2 = o3d.geometry.PointCloud()
                pcd2.points = o3d.utility.Vector3dVector(points2)
                pcd2.paint_uniform_color([0, 1, 0])

                o3d.visualization.draw_geometries([pcd1, pcd2])


        visualize(hands_to_base, eyes_to_world, R_eye_to_hand, t_eye_to_hand, t_world_to_base)

        #R_eye_to_hand*s*t_world_to_eye + t_eye_to_hand = R_base_to_hand*t_world_to_base + t_base_to_hand
        for iter in range(100):
            H = np.zeros((9, 9)).astype(np.float64)
            b = np.zeros((9, 1)).astype(np.float64)
            translation_rmse = 0
            errors = []
            for id in hands_to_base.keys():
                R_eye_to_world,t_eye_to_world = eyes_to_world[id]
                t_world_to_eye = -R_eye_to_world.transpose()@t_eye_to_world

                R_hand_to_base,t_hand_to_base = hands_to_base[id]
                R_base_to_hand,t_base_to_hand = R_hand_to_base.transpose(),-R_hand_to_base.transpose()@t_hand_to_base

                error = R_eye_to_hand@t_world_to_eye + t_eye_to_hand - R_base_to_hand@t_world_to_base - t_base_to_hand
                translation_error = np.linalg.norm(error)

                errors.append([id, translation_error])
                translation_rmse += translation_error*translation_error

                J = np.zeros((3, 9))
                J_err_R_eye_to_hand = -R_eye_to_hand@skworld_to_eye_symmetric_matrix(t_world_to_eye)
                J_err_t_eye_to_hand = np.eye(3)
                J_err_t_world_to_base = -R_base_to_hand
                J[:,:3] = J_err_R_eye_to_hand
                J[:,3:6] = J_err_t_eye_to_hand
                J[:,6:] = J_err_t_world_to_base

                H += J.transpose().dot(J)
                b += J.transpose()@error

            print('iter: ',iter,
                  ' translation rmse: ',math.sqrt(translation_rmse/len(images)),'(mm) ')

            last_translation_rmse = math.sqrt(translation_rmse/len(images))
            x = np.linalg.inv(H).dot(b)
            if np.linalg.norm(x)<1E-8:
                break
            step_R_eye_to_hand = -x[:3]
            R_eye_to_hand = R_eye_to_hand@exponentialMap(step_R_eye_to_hand)

            step_t_eye_to_hand = -x[3:6]
            t_eye_to_hand += step_t_eye_to_hand

            step_t_world_to_base = -x[6:9]
            t_world_to_base += step_t_world_to_base


        visualize(hands_to_base, eyes_to_world, R_eye_to_hand, t_eye_to_hand, t_world_to_base)
        
        #estimate world to base rotation
        def logarithmMap(R):
            v,_ = cv2.Rodrigues(R)
            return v
        log_R_world_to_base = np.zeros((3,1))
        
        for id in hands_to_base.keys():
            R_eye_to_world, t_eye_to_world = eyes_to_world[id]
            R_hand_to_base, t_hand_to_base = hands_to_base[id]
            R_world_to_base = R_hand_to_base@R_eye_to_hand@R_eye_to_world.transpose()
            log_R_world_to_base += logarithmMap(R_world_to_base)
        log_R_world_to_base /= len(hands_to_base.keys())
        R_world_to_base = exponentialMap(log_R_world_to_base)
        #print("world to base rotation:",R_world_to_base)
        
        Logging.info("rotation w:",R_world_to_base)
        Logging.info("translation w:",t_world_to_base)
        Logging.info("rotation:",R_eye_to_hand)
        Logging.info("translation:",t_eye_to_hand)

        calibrated = {"R":R_eye_to_hand,
                      "T":t_eye_to_hand,
                      "K":mtx,
                      "D":dist,
                      "W":R_world_to_base,
                      "Q":t_world_to_base}
        
        roi.update({"R":R_eye_to_hand.tolist()})
        roi.update({"T":t_eye_to_hand.tolist()})
        roi.update({"K":mtx.tolist()})
        roi.update({"D":dist.tolist()})
        roi.update({"W":R_world_to_base.tolist()})
        roi.update({"Q":t_world_to_base.tolist()})
        
        start.ActionUi.reload_detection()
        start.ActionUi.save_profile()

def take_data(ROW = 8, COL = 5):
    ret, roi = start.Profile.get_selected_roi()
    if ret:
        start.Com.select_user_coordinate_system(0) # World Frame Selected
        start.Com.select_tool_coordinate_system(0) # End Flage Selected 
        time.sleep(0.1)
            
        calibration_dataset_path = start.Profile.get_profile_folder_path() / 'calibration_data' / roi['name']
        
        current_num = start.Profile.get_next_jpg_number(roi['name'])
        
        filename = str(current_num) + ".png"
        
        try:
            with open( calibration_dataset_path / 'calibration_dataset.json','r') as file:
                calibration_dataset = json.load(file)
        except:
            calibration_dataset = {"data":[]}
            
        raw_frame = list(start.Cam.get_image().values())[start.MainUi.cam_pos]
        
        gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (COL, ROW), None)
        start.MainUi.update_result_frame(cv2.drawChessboardCorners(raw_frame.copy(), (COL, ROW), corners, ret))
        if ret == True:
        
            cv2.imwrite(os.path.join(calibration_dataset_path,str(current_num)+".png") , raw_frame)

            tcp_data = get_tcp()
            
            temp_record = {}
            temp_record.update({"jx":tcp_data[0], "jy":tcp_data[1], "jz":tcp_data[2],
                                "rx":tcp_data[3], "ry":tcp_data[4], "rz":tcp_data[5],

                                "record_num":current_num ,
                                "image":filename})
            
            calibration_dataset['data'].append(temp_record)
            
            # Serializing json
            json_object = json.dumps(calibration_dataset, indent=4)

            # Writing to sample.json
            with open(calibration_dataset_path / "calibration_dataset.json", "w") as outfile:
                outfile.write(json_object)
            messagebox.showinfo("Collect Data", "Data Collected Successfully")
        else:
            messagebox.showwarning("Warning", "No Chessboard Found")

def take_data_2d(ROW = 8, COL = 5):
    ret, roi = start.Profile.get_selected_roi()
    if ret:
        start.Com.select_user_coordinate_system(0) # World Frame Selected
        start.Com.select_tool_coordinate_system(0) # End Flage Selected 
        time.sleep(0.1)
            
        calibration_dataset_path = start.Profile.get_profile_folder_path() / 'calibration_data' / roi['name']
        
        current_num = start.Profile.get_next_jpg_number(roi['name'])
        
        filename = str(current_num) + ".png"
        
        try:
            with open( calibration_dataset_path / 'calibration_dataset.json','r') as file:
                calibration_dataset = json.load(file)
        except:
            calibration_dataset = {"data":[]}
            
        raw_frame = list(start.Cam.get_image().values())[start.MainUi.cam_pos]
        
        

        a = Match.match_rotate(roi.get('img'),raw_frame.copy())
        if a:
            frame, cropped, [x_center, y_center], angle , dst, xy_dst = a
            start.MainUi.update_result_frame(frame)
            cv2.imwrite(os.path.join(calibration_dataset_path,str(current_num)+".png") , raw_frame)
            cv2.imwrite(os.path.join(calibration_dataset_path,"template"+".bmp") , roi.get('img'))
            tcp_data = get_tcp()
            
            temp_record = {}
            temp_record.update({"jx":tcp_data[0], "jy":tcp_data[1], "jz":tcp_data[2],
                                "rx":tcp_data[3], "ry":tcp_data[4], "rz":tcp_data[5],
                                "points":dst.tolist(),
                                "record_num":current_num ,
                                "image":filename})
            
            calibration_dataset['data'].append(temp_record)
            
            # Serializing json
            json_object = json.dumps(calibration_dataset, indent=4)

            # Writing to sample.json
            with open(calibration_dataset_path / "calibration_dataset.json", "w") as outfile:
                outfile.write(json_object)
            messagebox.showinfo("Collect Data", "Data Collected Successfully")
