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
import json
import copy
import cv2
from PIL import Image, ImageTk
import traceback 
import time
from threading import Thread
from FMCV.Cv import Cv
from FMCV import Logging
import numpy as np


current_step = 0 
detected_step = 0

results = []

result_frame = []

repeat = 1

flag_reset = True

flag_reload = True

selected_source = None

barcode = ""

started = False

last_roi_state = True
last_overall_state = True
continuous_roi_fail_count = 0
continuous_overall_fail_count = 0 

cycle_start_time =  time.time()


def init(s):
    global self, start
    global live, off_live
    self = s 
    start = s

def is_overall_pass():
    is_pass = True
    try:
        for src_n, src in enumerate(self.Main.results):
            for step_n, step in enumerate(self.Main.results[src_n]):
                for roi_n, roi in enumerate(self.Main.results[src_n][step_n]):
                    roi_pass = roi.get('PASS')
                    if roi_pass is True:
                        is_pass = True
                    else :
                        is_pass = False
                        raise StopIteration
    except StopIteration:
        pass
    return is_pass

def is_step_pass(in_step):
    is_pass = True
    try:
        for src_n, src in enumerate(self.Main.results):
            for roi_n, roi in enumerate(self.Main.results[src_n][in_step]):
                roi_pass = roi.get('PASS')
                if roi_pass is True:
                    is_pass = True
                else :
                    is_pass = False
                    raise StopIteration
    except StopIteration:
        pass
    return is_pass

# Only uses by edit
def set_step(step):
    global current_step
    global detected_step
    detected_step = step
    current_step = step

def detect(step = None ,SN = "", in_frames = None):
    global flag_reset, flag_reload, started
    global current_step
    global detected_step
    global barcode
    global cycle_start_time
    global last_overall_state, continuous_overall_fail_count, continuous_roi_fail_count
    
    global platform_non_stop
    
    if start.Config.non_stop and start.Config.platform_model != "NONE":
        platform_non_stop = True
    else:
        platform_non_stop = False
    
    # Disabled
    # if start.config.CONTROL.enable_move: 
        # platform_non_stop = True    
        # for roi_n, roi_result in enumerate(results[0][current_step]):
            # if roi_result['type'] == 'MOVE':
                # if roi_result.get("pos_r_xyz") is not None:
                    # if not roi_result.get("absolute_joint"):
                        # start.Com.select_user_coordinate_system(10)
                        # start.Com.select_tool_coordinate_system(10)
                        # start.Com.to_tcp([0,0,0,0,0,0])
                        # time.sleep(0.1)
                        # start.pub("com/robot/to_tcp",
                                        # roi_result.get("pos_r_xyz"),
                                        # relative_move=False,
                                        # speed=roi_result.get("speed"), 
                                        # accel=roi_result.get('accelerate'))
                    # else:
                        # start.Com.select_user_coordinate_system(0)
                        # start.Com.select_tool_coordinate_system(0)
                        # start.Com.to_tcp([0,0,0,0,0,0])
                        # time.sleep(0.1)
                        # start.pub("com/robot/to_tcp",
                                        # roi_result.get("pos_r_xyz"),
                                        # relative_move=False,
                                        # speed=roi_result.get("speed"), 
                                        # accel=roi_result.get('accelerate'))
        #self.MainUi.refresh()
    #else:
        #platform_non_stop = False
        
        
    if current_step == 0:
        cycle_start_time =  time.time()

    if step is not None:
        print("current_step to {}".format(step))
        current_step = step
    
    if flag_reset:
        print("reseting steps")
        reset()
        
    if flag_reload:
        print("detection reload")
        reload()   

    started = True
    
    detected_step = current_step
    
    if self.MainUi.barcode_entry.get() != "":
        print(self.MainUi.barcode_entry.get())
        barcode = self.MainUi.barcode_entry.get()
    
    if SN != "":
        barcode = SN
        print(f'Incoming {barcode}')

    move_platform_to_position(current_step)

    detect_next_results = start_detect(step, in_frames)
    
    #Disabled
    # if start.config.CONTROL.enable_move: 
        # platform_non_stop = True    
        # for roi_n, roi_result in enumerate(results[0][current_step]):
            # if roi_result['type'] == '2D':
                # if roi_result['PASS']:
                    # if roi_result.get('angle') >-999 and roi_result.get('offset_x_mm') > float('-inf') and roi_result.get("offset_y_mm") > float('-inf'):
                        # start.Com.select_user_coordinate_system(10)
                        # start.Com.select_tool_coordinate_system(10)
                        # start.Com.to_tcp([0,0,0,0,0,0])
                        # time.sleep(0.1)
                        
                        # tcp = [roi_result.get('offset_x_mm'),roi_result.get("offset_y_mm"),0,0,0,roi_result.get('angle')]
                        # start.Com.to_tcp(tcp,relative_move=True)
                        
                        # time.sleep(0.1)
                        
                        # start.Com.select_user_coordinate_system(0)
                        # start.Com.select_tool_coordinate_system(0)
                        # start.Com.to_tcp([0,0,0,0,0,0])
                        # time.sleep(0.1)
                        
                        # current_pos = start.Com.get_tcp()
                        
                        # eye_to_hand_R = np.array(roi_result.get('R'))
                        # eye_to_hand_t = np.array(roi_result.get('T'))
                        
                        # user_coordinate_offset = start.Calibrate3d.getCameraPoseGivenHandPose(current_pos, eye_to_hand_R, eye_to_hand_t)
                        # start.Com.set_user_coordinate_system(user_coordinate_offset,10,"FMCV Board")
                        # start.Com.to_tcp([0,0,0,0,0,0])
                        # time.sleep(0.1)
                        # self.MainUi.refresh()
                # else:
                    # platform_non_stop = False
    #else:
    #    platform_non_stop = False
    
    print(f"detect_next_results {detect_next_results}")
    
    if detect_next_results:
        next_step() #current_step increment
    

        #print(">>{} {}".format(continuous_roi_fail_count, continuous_overall_fail_count))
    
#current_step increment logic
def next_step():
    global last_overall_state, continuous_overall_fail_count   
    global detected_step
    global started
    global current_step
    global flag_reset
    global flag_reload
    global self
    global barcode
    
    global platform_non_stop

    if flag_reset:
        print("reseting steps")
        reset()
        
    if flag_reload:
        print("detection reload")
        reload()   
        
    started = True    
    
    detected_step = current_step
    
    current_step = current_step + 1

    if current_step > len(results[0]) - 1: # Last step

        current_step = 0
        
        update_total()
        
        start.MainUi.result_frame.set_overall_result(is_overall_pass())
        
        #Continueous fail detection    
        overall_state = is_overall_pass()
        if not overall_state:
            self.Com.failed()
            # 3x Failed Counter
            if last_overall_state == False:
                continuous_overall_fail_count += 1
            else:
                continuous_overall_fail_count = 0
        last_overall_state = overall_state
        Logging.info(f'the barcode is {barcode}')
        self.Log.write_log()
                
        self.MainUi_Refresh.refresh_main_ui()
        platform_non_stop = False
        
        if start.config.MODE.show_results:
            start.pub("ui/results/show")
            start.pub("ui/results/set_barcode",barcode)
            
        barcode = ""
        self.MainUi.barcode_entry.delete(0, 'end')
        
    Logging.info("current_step is {}".format(current_step))

    # if platform_non_stop and start.config.CONTROL.enable_move:
        # self.MainUi_Refresh.refresh_main_ui()
        # self.MainUi.view.after(1, detect)
    # elif platform_non_stop:
        # self.MainUi_Refresh.refresh_main_ui()
        # self.MainUi.view.after(350, detect)
    if platform_non_stop:
        self.MainUi_Refresh.refresh_main_ui()
        self.MainUi.view.after(350, detect)
        
def reset():
    global current_step, results, result_frame, barcode, started, flag_reset
    
    if started:
        if self.Config.reset_log:
            self.Log.write_log()
    
    started = False
    current_step = 0    
    barcode = ""
    
    start.MainUi.barcode_entry.delete(0, 'end')
    start.MainUi.result_frame.reset()
    
    results.clear()
    for src_n, src in enumerate(self.Profile.loaded_profile):
        results.append([])
        for step_n, step in enumerate(src):
            results[src_n].append([])
            for roi_n, roi in enumerate(step["roi"]):
                results[src_n][step_n].append(copy.deepcopy(roi))
            
    result_frame.clear()
    del result_frame
    result_frame = []
    for src_n, src in enumerate(self.Profile.loaded_profile):
        result_frame.append([])        
        for step_n, step in enumerate(src):
            result_frame[src_n].append({"":None})
    
    flag_reset = False
    
    print('Reset step!')

def reload():
    global current_step, results, result_frame, barcode, started, flag_reset, flag_reload
    
    results.clear()
    for src_n, src in enumerate(self.Profile.loaded_profile):
        results.append([])
        for step_n, step in enumerate(src):
            results[src_n].append([])
            for roi_n, roi in enumerate(step["roi"]):
                results[src_n][step_n].append(copy.deepcopy(roi))
            
    result_frame.clear()
    for src_n, src in enumerate(self.Profile.loaded_profile):
        result_frame.append([])        
        for step_n, step in enumerate(src):
            result_frame[src_n].append({"":None})
            
    flag_reload = False 

def get_results_with_result_image_by_name(name):
    ret = False
    result_rois = []
    for src_n, src in enumerate(results):
        for step_n, step in enumerate(src):
            for roi_n, roi in enumerate(step):
                if roi['name'] == name and roi.get('result_image') is not None:
                    ret = True
                    result_rois.append(roi)
    return ret, result_rois
            

def start_detect(step = None, in_frames = None):
    global current_step, results, result_frame, barcode
    global last_roi_state, continuous_roi_fail_count
    global continuous_overall_fail_count, last_overall_state
    
    start.MainUi.result_frame.set_running()
    
    if in_frames is None:
        frames = self.Cam.get_image()
    else:
        frames = in_frames
        
    is_pass = True
    for src_n, src in enumerate(results):
        try:
            if list(frames.values())[src_n] is not None:
                frame = copy.deepcopy(list(frames.values())[src_n])
                result_frame[src_n][current_step].update({"":frame})
                
                #print(frame.shape)
                temp_frms = {}
                for roi_n, roi_result in enumerate(results[src_n][current_step]):
                    
                    frm = temp_frms.get(roi_result['rotate'])
                    if frm is None:
                        temp_frms.update({roi_result['rotate'] : Cv.get_rotate(roi_result['rotate'], frame)})
                        result_frame[src_n][current_step].update({roi_result['rotate'] : temp_frms[roi_result['rotate']]})
                        frm = result_frame[src_n][current_step][roi_result['rotate']]
                        
                    if roi_result.get('source') is not None:
                        if roi_result.get('source') != "":
                            ret, result_rois = get_results_with_result_image_by_name(roi_result.get('source'))
                            if ret:
                                frm = Cv.get_rotate(roi_result['rotate'], result_rois[0].get('result_image'))
                                roi_result.update({'source_image':frm})
                                
                    roi_result = self.Process.execute(frm ,roi_result, src_n, current_step, roi_n)

                    if not roi_result.get('PASS'):
                        is_pass = False
                        
                        #Continueous fail detection    
                        if last_roi_state == False:
                            continuous_roi_fail_count += 1
                        else:
                            continuous_roi_fail_count = 0
                    last_roi_state = roi_result.get('PASS')
                    
                    if roi_result['type'] == "QR":
                        barcode = str(roi_result.get("CODE"))
                        
        except:
            is_pass = False
            traceback.print_exc()
    
    if start.Config.show_running == False:
        start.MainUi.result_frame.set_result(is_pass)
    
    failed_3x = False    
    if start.Config.alarm_if_fail_3x:
        if continuous_roi_fail_count >= 2 :
            print("ROI Failed equal or more then 3 times")
                
        if continuous_overall_fail_count >= 2:
            print("Overall Failed equal or more then 3 times")
            
            if not start.MainUi.ask_reset_continuous_fail_alarm():
                last_overall_state = True
                continuous_overall_fail_count = 0
                failed_3x = True

    if is_pass and step is None and in_frames is None: 
        print("Com go next")
        self.Com.go_next()
    elif failed_3x:
        self.Com.failed()
        #self.Com.alarm()
    elif self.Config.non_stop and in_frames is None:
        print("Non Stop Com go next")
        is_pass = True
        self.Com.go_next()        
    elif not is_pass and step is None and in_frames is None:
        print("Com Failed")
        self.Com.failed()
        self.Log.write_log()

        if len(results[0]) == 1: # this is a hack for single step application on fail condition to proceed next step when fail
            is_pass = True
    elif in_frames is not None:
        is_pass = True #speed mode
        
    return is_pass    
 
def reset_total_count():
    start.Config.class_total.update({"PASS":0,"FAIL":0})
    start.Config.write_total()
    
def update_total():
    total_pass = start.Config.class_total.get("PASS")
    total_fail = start.Config.class_total.get("FAIL")
    
    if is_overall_pass():
        total_pass = total_pass + 1
    else :
        total_fail = total_fail + 1
    start.Config.class_total.update({"PASS":total_pass,"FAIL":total_fail})


def init_moving_platform():
    """Create instance for moving platform and try to connect to it"""
    print("Initialize moving platform")
    try:
        # Create instance for moving platform and its current position
        start.MovingPlatform = start.Platform.platform_factory(model=start.Config.config['PLATFORM']['model'], feedback_callback=None)
        #start.MovingPlatform = start.Platform.platform_factory(model=start.Config.config['PLATFORM']['model'], feedback_callback=moving_platform_status_feedback_handler)
        self.current_platform_position = start.CartesianPosition()

        # start the auto connect/reconnect thread
        if(start.MovingPlatform is not None):
            self.is_platform_start = True
            self.moving_platform_auto_connect_thread = Thread(target=moving_platform_auto_connect_task, daemon=True)
            self.moving_platform_auto_connect_thread.start()
    except:
        print(f"Initialize moving platform {start.Config.config['PLATFORM']['model']} failed")
        traceback.print_exc()
        pass

def uninit_moving_platform():
    """Stop the autoconnect thread and it will disconnect also"""
    self.is_platform_start = False
    self.moving_platform_auto_connect_thread.join()


def moving_platform_auto_connect_task():
    """Task to auto connect to moving platform's hardware if it is disconnected"""
    print("Start moving platform auto connect task")
    while True:
        if (self.is_platform_start == False):
            # disconnect from the hardware
            start.MovingPlatform.disconnect()
            print(f"Disconnect from {start.Config.config['PLATFORM']['model']}")
            self.MainUi.update_platform_status(is_connected=True, mode=start.MovingPlatform.operating_mode())

            # break the while loop and end the thread
            break

        try:
            if ((start.MovingPlatform.get_is_connected() == False) and self.is_platform_start):
                # try to connect
                print(f"Try to connect to {start.Config.config['PLATFORM']['ip_address']}")
                start.MovingPlatform.connect(start.Config.config['PLATFORM']['ip_address'])

            # check it every 2 seconds
            time.sleep(2)
        except:
            traceback.print_exc()
            print(f"Moving Platform {start.Config.config['PLATFORM']['model']} Connection Error!")

    print("Moving Platform Auto Connect Task Ended")

def moving_platform_status_feedback_handler(position_data):
    # todo: suppose the position is feedback by the callback function, not from the instance
    try:
        self.current_platform_position.set_x(start.MovingPlatform.x())
        self.current_platform_position.set_y(start.MovingPlatform.y())
        self.current_platform_position.set_z(start.MovingPlatform.z())
        self.current_platform_position.set_roll(start.MovingPlatform.roll())

        # call the function in UI module to update the position to UI
        self.MainUi.update_platform_status(is_connected=True, mode=start.MovingPlatform.operating_mode())
    except:
        traceback.print_exc()


def move_platform_to_position(step_index):
    # move the platform to its position
    try:
        if self.config.PLATFORM.model.casefold() == "Dobot.MG400".casefold():
            # get selected position from profile
            self.platform_position = self.Profile.loaded_profile[self.MainUi.cam_pos][step_index]["platform"]
            Logging.info(f"Position: {self.platform_position}")

            x = self.platform_position['x']
            y = self.platform_position['y']
            z = self.platform_position['z']
            roll = self.platform_position['roll']

            if (self.MovingPlatform is not None):
                if (self.MovingPlatform.get_is_connected() == True):
                    self.MovingPlatform.move_to_point_sync(x, y, z, roll)
    except Exception as e:
        Logging.error(traceback.format_exc())
        Logging.error("Error during reset platform position")
        
        
import queue
main_queue = queue.Queue()

def input_trigger():
    global main_queue, start
    
    while True:
        try:
            obj, msg, frames = main_queue.get()
            print(msg)                
            if msg == "T":
                print("Callback Triggered with frames")
                start.ActionUi.detect_with_frames(in_frames = frames)
            elif msg == "RESET":
                print("Callback RESET")
                start.ActionUi.reset_detection_step()
            elif msg == "VS_MES":
                print("VS MES Callback Triggered")
                start.ActionUi.detect(SN=obj.barcode)
                print(obj)
            elif msg.startswith('T,'):
                step = msg.split(",")[1]
                print(f"Callback Trigger with Step : {step}")
                if int(step)-1 >= 0:
                    start.ActionUi.detect_step(int(step)-1)
            elif msg == "Hello World":
                time.sleep(1)
                start.Com.respond("Echo done")
        except:
            traceback.print_exc()

trigger_thread = Thread(target=input_trigger, daemon=True)
trigger_thread.start()