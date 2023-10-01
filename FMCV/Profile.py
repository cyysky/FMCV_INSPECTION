# #==========================================================================
# import hashlib
# from FMCV.peace import Peace,license
# lic = license()['Vision']
# if (hashlib.sha3_256((lic+lic).encode('utf-8')).hexdigest()!=Peace(lic)):
    # # https://stackoverflow.com/questions/9555133/e-printstacktrace-equivalent-in-python
    # #traceback.print_exc()
    # # https://stackoverflow.com/questions/73663/terminating-a-python-script
    # import os
    # import sys
    # os._exit(0)
    # sys.exit(0)
    # raise SystemExit  
    # 1/0      
    # pass
# #==========================================================================

import os
from os.path import join
import json
import traceback
import cv2
import copy
import base64
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from FMCV.Cv import Cv


name = "default"

profile = []

loaded_profile = []

flag_save = False

def init(s=None):
    global start
    if s is not None:
        start = s 
    if start.args.profile is not None:
        start.Config.profile = start.args.profile #Manually override profile name by Command Line -p --profile
    read(start.Config.profile)
    
    global Logging
    Logging = start.Logging

def select_profile(in_name):
    if start.Users.login_user():
        read(in_name)

def read(in_name):
    global profile, name
    name = in_name
    try:
        os.makedirs(join("Profile",name,"images"), exist_ok=True)        
        os.makedirs(join("Profile",name), exist_ok=True)
        with open(join("Profile",name,"profile.json"), "r") as f:
            profile = json.load(f)
        load()
    except:
        traceback.print_exc()
        
        profile = []
        
        profile.append([]) # src
        
        profile[0].append({"platform": {"x": 350, "y": 0, "z": 0, "roll": 0}, "roi": []})  # step
        
        # profile[0][0].append([]) # roi
        
        # profile[0][0][0].update({})
        
        print(json.dumps(profile , indent = 4))
        load()
        write()

def write():
    global profile, flag_save
    if start.Users.login_admin():
        #print(loaded_profile)
        profile.clear()
        for src_n, src in enumerate(loaded_profile):
            profile.append([])
            for step_n, step in enumerate(src):
                #
                profile[src_n].append([])
                profile[src_n][step_n] = {}
                profile[src_n][step_n]["platform"] = step["platform"]
                profile[src_n][step_n]["roi"] = []
                for roi_n, roi in enumerate(step["roi"]):
                    #print(self.Util.without_keys(roi,{"img"})) #debug use
                    profile[src_n][step_n]["roi"].append(start.Util.without_keys(roi,{"img",'mask'}))
                    #if roi.get('img') is not None: # moved to UI
                    #    profile[src_n][step_n][roi_n].update({"image":base64.b64encode(cv2.imencode('.png',roi['img'])[1]).decode()})
                    
        file_name = start.Util.utc_to_local(datetime.utcnow()).strftime('%Y-%m-%d_%H%M%S_%f')[:-3] + "_profile.json"
        try :
            os.makedirs(join("Profile",name,"Profile_Backup"), exist_ok=True)     
            os.replace(join("Profile",name,"profile.json"),join("Profile",name,"Profile_Backup",file_name))
        except:
            print("no rename needed")
        with open(join("Profile",name,"profile.json"), "w") as outfile:
            outfile.write(json.dumps(profile , indent = 4))
            print(join("Profile",name,"profile.json"))
            flag_save = False

def get_roi_by_name(name):
    rois = []
    ret = False
    for src_n, src in enumerate(start.Profile.loaded_profile):
        for step_n, step in enumerate(src):
            for roi_n, roi in enumerate(step["roi"]):
                if roi['name'] == name:
                    ret = True
                    rois.append(roi)
    return ret, rois
            
def load():
    global profile, loaded_profile, name
    
    get_image_base_path()
    
    loaded_profile.clear()
    loaded_profile
    for src_n, src in enumerate(profile):
        loaded_profile.append([])
        for step_n, step in enumerate(src):
            loaded_profile[src_n].append([])
            # print("Step:")
            # print(json.dumps(step["platform"], indent=4))
            # # extract platform param
            # loaded_profile[src_n][step_n] = step["platform"]
            loaded_profile[src_n][step_n] = {}
            
            if isinstance(step, list):# post platform profile compatible 
                loaded_profile[src_n][step_n]["platform"] = {"x": 350, "y": 0, "z": 0, "roll": 0}
                loaded_profile[src_n][step_n]["roi"] = []
                
                for roi_n, roi in enumerate(step): 
                    loaded_profile[src_n][step_n]["roi"].append(roi)
                    if roi.get('image') is not None:
                        loaded_profile[src_n][step_n]["roi"][roi_n].update({"img":cv2.imdecode(np.frombuffer(base64.b64decode(roi["image"]), dtype=np.uint8),flags=cv2.IMREAD_COLOR)})
                    if roi.get('mask_64') is not None:
                        loaded_profile[src_n][step_n]["roi"][roi_n].update({"mask":cv2.imdecode(np.frombuffer(base64.b64decode(roi["mask_64"]), dtype=np.uint8),flags=cv2.IMREAD_GRAYSCALE)})
            if isinstance(step, dict): # new profile structures
                loaded_profile[src_n][step_n]["platform"] = step["platform"]
                loaded_profile[src_n][step_n]["roi"] = []
                for roi_n, roi in enumerate(step["roi"]):
                    # each step include platform position and multiple ROIs
                    loaded_profile[src_n][step_n]["roi"].append(roi)
                    #print(json.dumps(roi, indent=4))
                    if roi.get('image') is not None:
                        loaded_profile[src_n][step_n]["roi"][roi_n].update({"img":cv2.imdecode(np.frombuffer(base64.b64decode(roi["image"]), dtype=np.uint8),flags=cv2.IMREAD_COLOR)})
                    if roi.get('mask_64') is not None:
                        loaded_profile[src_n][step_n]["roi"][roi_n].update({"mask":cv2.imdecode(np.frombuffer(base64.b64decode(roi["mask_64"]), dtype=np.uint8),flags=cv2.IMREAD_GRAYSCALE)})
                    if roi.get('rotate') is None:
                        loaded_profile[src_n][step_n]["roi"][roi_n].update({"rotate":""})
                        
                        
def balance_step():
    global loaded_profile
    # Find the length of the longest list
    max_len = max(len(lst) for lst in loaded_profile)

    # Balance the length of each list
    for lst in loaded_profile:
        while len(lst) < max_len:
            lst.append({"platform": {"x": 350, "y": 0, "z": 0, "roll": 0}, "roi": []})  # add values to the end of the list

                        
                        
def add_source(src_n):
    if start.Users.login_admin():
        print(src_n)
        global loaded_profile
        loaded_profile.insert(src_n + 1, [])
        loaded_profile[src_n + 1].append({"platform": {"x": 350, "y": 0, "z": 0, "roll": 0}, "roi": []})
        balance_step()
    
def remove_source(src_n):
    if start.Users.login_admin():
        global loaded_profile
        loaded_profile.pop(src_n)

def add_step(src_n, step_n):
    if start.Users.login_admin():
        print(step_n)
        global loaded_profile
        loaded_profile[src_n].insert(step_n + 1, {"platform": {"x": 350, "y": 0, "z": 0, "roll": 0}, "roi": []})
        balance_step()
    
def insert_step(src_n, step_n):
    if start.Users.login_admin():
        print(step_n)
        global loaded_profile
        loaded_profile[src_n].insert(step_n, {"platform": {"x": 350, "y": 0, "z": 0, "roll": 0}, "roi": []})
        balance_step()

def move_step(src_n, step_n , new_step_n):
    if start.Users.login_admin():
        print(f"move stop from cam:{src_n} step:{step_n} to step:{new_step_n}")
        global loaded_profile
        loaded_profile[src_n].insert(new_step_n,loaded_profile[src_n].pop(step_n))
        

def duplicate_step(src_n, step_n , new_step_n):
    if start.Users.login_admin():
        print(f"move stop from cam:{src_n} step:{step_n} to step:{new_step_n}")
        global loaded_profile
        loaded_profile[src_n].insert(new_step_n,copy.deepcopy(loaded_profile[src_n][step_n]))
        balance_step()

def remove_step(src_n_na, step_n):
    if start.Users.login_admin():
        global loaded_profile
        for src_n, src in enumerate(loaded_profile):
            loaded_profile[src_n].pop(step_n)

    
def find_roi_index(in_roi):
    for i, sources in enumerate(loaded_profile):
        for j, steps in enumerate(sources):
            if 'roi' in steps:
                rois = steps['roi']
                for k, roi in enumerate(rois):
                    if roi is in_roi:
                        return i,j,k
    return None

def select_roi(in_roi):
    roi_indexs = find_roi_index(in_roi)
    if roi_indexs is None:
        return
        
    i, j, k = roi_indexs
    
    start.log.info(f"ROI's index source={i} step={j} roi={k}")
    
    start.MainUi.cam_pos = i
    start.MainUi.cmb_pos = j
    start.Main.detected_step = j # MainUi_Refresh.refresh_main_ui() follow detected_step to refresh
    start.MainUi.roi_index = k
    
    start.MainUi_Refresh.refresh_main_ui()

def paste_roi(roi):
    global loaded_profile
    if start.Users.login_admin():
        loaded_profile[start.MainUi.cam_pos][start.MainUi.cmb_pos]["roi"].append(copy.deepcopy(roi))
        start.MainUi_Refresh.refresh_main_ui()
        
def add_roi(x1=30,x2=300,y1=30,y2=300):
    global loaded_profile
    if start.Users.login_admin():
        image_height = 270
        image_width = 270
        number_of_color_channels = 3
        color = (255,255,255)
        pixel_array = np.full((image_height, image_width, number_of_color_channels), color, dtype=np.uint8)
    
        roi = {   
            "name":f"{start.MainUi.roi_entry.get()}",
             "x1":x1,
             "x2":x2 ,
             "y1":y1,
             "y2":y2,
             "img":pixel_array,
             "margin" : 0,   #"smooth" : 0,
             "rotate": "",   #"precheck" : False, "pre_value" : 0.55,
             "type":"AI",    #"class":"",
             "minimal":0.95, #"return":True
                    }
        loaded_profile[start.MainUi.cam_pos][start.MainUi.cmb_pos]["roi"].insert(start.MainUi.roi_index+1,roi)
        

def update_roi(src_n, step_n, roi_n, roi):
    if start.Users.login_admin():
        global loaded_profile
        loaded_profile[src_n][step_n]["roi"][roi_n].update(roi)
 
def move_roi(src_n, step_n, roi_n, new_roi_n):  
    if start.Users.login_admin():
        print(f"move roi from cam:{src_n} step:{step_n} roi:{roi_n} to roi:{new_roi_n}")
        global loaded_profile
        loaded_profile[src_n][step_n]["roi"].insert(new_roi_n, loaded_profile[src_n][step_n]["roi"].pop(roi_n))
        
def remove_roi(src_n, step_n,roi_n):
    if start.Users.login_admin():
        global loaded_profile
        loaded_profile[src_n][step_n]["roi"].pop(roi_n)

def update_roi(src_n, step_n,roi_n,pair):
    if start.Users.login_admin():
        global loaded_profile
        loaded_profile[src_n][step_n]["roi"][roi_n].update(pair)
        
        
def get_selected_roi():
    roi_index = start.MainUi.roi_index
    roi = None
    has_roi = False
    if roi_index > -1:
        roi = start.Profile.loaded_profile[start.MainUi.cam_pos][start.MainUi.cmb_pos]["roi"][roi_index]
        has_roi = True
    return has_roi, roi


def get_selected_roi_frame():
    frame = None
    ret, roi = get_selected_roi()
    if not ret:
        return ret, frame
        
    frame = list(start.Cam.get_image().values())[start.MainUi.cam_pos]
    frame = Cv.get_rotate(roi['rotate'],frame)
    has_frame = True
    return has_frame, frame

def write_image(image_folder_name, cropped):
    if start.Users.login_user():
        file_name = start.Util.utc_to_local(datetime.utcnow()).strftime('%Y-%m-%d_%H%M%S_%f')[:-3]
        if start.Config.list_profile_images_folder:
            os.makedirs(join("Profile", name, "images", image_folder_name), exist_ok=True)
        os.makedirs(join(base_path, image_folder_name), exist_ok=True)
        cv2.imwrite(join(base_path, image_folder_name, file_name+".png"), cropped)
        return join(base_path, image_folder_name, file_name+".png")
    
def write_ano_image(cropped):
    if start.Users.login_user():
        file_name = start.Util.utc_to_local(datetime.utcnow()).strftime('%Y-%m-%d_%H%M%S_%f')[:-3]
        start.ANO.DATASET_PATH.mkdir(parents=True, exist_ok=True)
        write_path = start.ANO.DATASET_PATH / (file_name+".png")
        cv2.imwrite(write_path.as_posix(), cropped)
        return write_path.as_posix()


def create_image_folder(image_folder_name):
    os.makedirs(join(base_path,image_folder_name), exist_ok=True)

def write_ai_suggestion_image(classname, filename,image):
    suggestion_image_path = get_profile_folder_path() / 'AI_suggestion_image' / classname 
    suggestion_image_path.mkdir(parents=True, exist_ok=True)
    
    cv2.imwrite(str(suggestion_image_path / filename), image)


def get_image_folders_list():
    if start.Config.list_profile_images_folder:
        return next(os.walk(Path("Profile", name, "images")))[1]
    else:
        return next(os.walk(base_path))[1]
    
def get_image_folder_path(folder_name):
    os.makedirs(join(base_path,folder_name), exist_ok=True)
    return join(base_path,folder_name)
    
def get_image_base_path():
    global base_path
    base_path = Path("Profile", name, "images")
    if str(start.Config.model_path) != ".":
        base_path = start.Config.model_path / "images"
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path
    
def get_profile_folder_path():
    return Path("Profile", name)
    
import os
from pathlib import Path

def get_next_jpg_number(roi_name):
    search_path = get_profile_folder_path() / 'calibration_data' / roi_name
    search_path.mkdir(parents=True, exist_ok=True)
    file_type = '.png'

    def sort_key(file_path):
        file_name = os.path.basename(file_path)
        file_base, file_ext = os.path.splitext(file_name)
        return int(file_base), file_ext.lower()

    # Search for files in the defined path with the specified file type
    file_paths = [
        str(p) for p in Path(search_path).glob(f'**/*{file_type}') if p.is_file()
    ]

    # Sort the file paths by filename and type (extension)
    sorted_file_paths = sorted(file_paths, key=sort_key)
    largest_number = 0
    # Get the largest number in the list
    if sorted_file_paths:
        largest_number = int(os.path.splitext(os.path.basename(sorted_file_paths[-1]))[0])
        Logging.info(f"Largest number:" ,largest_number," and next number are",largest_number+1)
        largest_number = largest_number +1
    else:
        Logging.info("No files found with the specified file type.")
    return largest_number