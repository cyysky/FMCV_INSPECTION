from configparser import ConfigParser
from FMCV.ConfigObj import ConfigObj

import traceback

from configparser import ConfigParser
from FMCV.ConfigObj import ConfigObj
from FMCV import Logging
import traceback

from pathlib import Path

import os

import json

class FMCV_Config(ConfigObj):
    def __init__(self, start):

        self.BRAND = start.BRAND

        self.version = start.version
        
        self.start = start
        config = ConfigParser()
        
        config.add_section('PROFILE')
        config['PROFILE']['name'] = 'default'
        
        config.add_section('HOST')
        config['HOST']['dialog_port'] = '0'
        config['HOST']['broadcast_port'] = '0'
        
        config.add_section('CONTROL')
        config['CONTROL']['trigger_delay_seconds'] = '0'
        config['CONTROL']['comport'] = 'NONE'
        config['CONTROL']['modbus_ip'] = 'NONE'
        config['CONTROL']['modbus_port'] = '502'
        config['CONTROL']['modbus_type'] = 'NONE'
        config['CONTROL']['tcp_ip'] = 'NONE'
        config['CONTROL']['tcp_port'] = 'NONE'
        config['CONTROL']['tcp_type'] = 'NONE'
        #config['CONTROL']['enable_move'] = 'False'
        
        config.add_section('CONNECT')
        config['CONNECT']['mes_type'] = 'NONE'
        config['CONNECT']['mes_path'] = 'NONE'
        
        config.add_section('LIGHTING')
        config['LIGHTING']['red'] = "127"
        config['LIGHTING']['green'] = "127"
        config['LIGHTING']['blue'] = "127"
        
        config.add_section('CUDA')
        config["CUDA"]["cuda_visible_devices"] = "-1"
        
        config.add_section('AI')
        config['AI']['minimum'] = "0.6"
        config['AI']['model_path'] = ""
        config['AI']['list_profile_images_folder'] = "False"

        config.add_section('CNN')
        config['CNN']['mode'] = "Normal"
        config['CNN']['keep_ratio'] = "True"
        config['CNN']['auto_save'] = "False"
        config['CNN']['train_rotate'] = "10"
        config['CNN']['train_brightness'] = "0.4"
        config['CNN']['train_width_shift']= "0.2" # horizontal shift
        config['CNN']['train_height_shift']= "0.2" # vertical shift
        config['CNN']['train_zoom_range']= "0.3"
        config['CNN']['train_horizontal_flip']= "False"
        config['CNN']['train_vertical_flip'] = "False"
        config['CNN']['train_save_augmentation'] = "False"

        config.add_section('VISION')
        config['VISION']['trigger_type'] = "NORMAL"
        
        config.add_section('CAMERA')
        config['CAMERA']['name'] = "Camera"

        config.add_section("PLATFORM")
        config['PLATFORM']['model'] = "NONE"
        config['PLATFORM']['ip_address'] = "192.168.1.6"
        
        config.add_section('MODE')
        config['MODE']['name'] = "ENGINEER"
        config['MODE']['spacebar_save_roi'] = "True"
        config['MODE']['non_stop'] = "N"
        config['MODE']['alarm_if_fail_3x'] = "False"
        config['MODE']['show_live'] = "True"
        config['MODE']['show_running'] = "True"
        config['MODE']['show_results'] = "True"
        config['MODE']['always_show_on_bottom'] = "False"
        config['MODE']['queue_frames'] = "False"
        config['MODE']['ai_detail'] = "False"
        
        
        config.add_section('LOG')    
        config['LOG']['type'] = "NORMAL" #KAIFA, FLEX, VS, NORMAL
        config['LOG']['images_path'] = "LOG/IMAGES"
        config['LOG']['results_path'] = "LOG/RESULTS"
        
        config['LOG']['mes_path'] = "LOG/MES" # Extra Path for KAIFA
        config['LOG']['tester_id'] = "ASVI_1" # Extra Attribute for KAIFA
        
        config['LOG']['backup_path'] = "LOG/BACKUP" # Backup Path
        
        config['LOG']['reset_log'] = "False"  # Write log while reset
        
        config.add_section('DEBUG')
        config['DEBUG']['level'] = "40"
        
        string_config = config
        
        super().__init__(string_config, "", "Backup", 'config.ini')
        
    def get_config_ready(self):
        
        print(f"\nFMCV_H Version {self.version}")
        config = self.config
        
        self.profile = config['PROFILE']['name']
        
        self.comport = config['CONTROL']['comport']
        self.modbus_ip = config['CONTROL']['modbus_ip']
        self.modbus_port = config['CONTROL']['modbus_port']
        self.modbus_type = config['CONTROL']['modbus_type']
        
        config['CONTROL']['trigger_delay_seconds'] = self.get_float(config['CONTROL']['trigger_delay_seconds'], fallback=0.0)
        
        #config['CONTROL']['enable_move'] = self.get_boolean(config['CONTROL']['enable_move'], fallback=False)
        
        config['HOST']['dialog_port'] = self.get_int(config['HOST']['dialog_port'], fallback=0)
        config['HOST']['broadcast_port'] = self.get_int(config['HOST']['broadcast_port'], fallback=0)
        
        
        self.r = config['LIGHTING']['red']
        self.g = config['LIGHTING']['green']
        self.b = config['LIGHTING']['blue']
        
        self.ai_minimum = self.get_float(self.config.AI.minimum, fallback=0.6)
        self.config.AI.minimum = self.ai_minimum
        
        self.model_path = Path(self.config.AI.model_path)
        
        self.ai_detail = self.get_boolean(self.config['MODE']["ai_detail"], fallback=False)
        self.config['MODE']["ai_detail"] = self.ai_detail 
        
        self.cnn_mode = config['CNN']['mode']     

        config['CNN']['keep_ratio'] = self.get_boolean(self.config['CNN']["keep_ratio"], fallback=True)
        
        self.list_profile_images_folder = self.get_boolean(config['AI']['list_profile_images_folder'], fallback=False)
        
        self.train_rotate = self.get_int(config['CNN']['train_rotate'], fallback=0)
        config['CNN']['train_rotate'] = self.train_rotate
        
        self.train_brightness = self.get_float(config['CNN']['train_brightness'] , fallback=0)      
        config['CNN']['train_brightness'] = self.train_brightness
        
        self.train_width_shift = self.get_float(config['CNN']['train_width_shift'], fallback=0)  # horizontal shift
        config['CNN']['train_width_shift'] =self.train_width_shift
        
        self.train_height_shift = self.get_float(config['CNN']['train_height_shift'], fallback=0) # vertical shift
        config['CNN']['train_height_shift'] = self.train_height_shift
        
        self.train_zoom_range = self.get_float(config['CNN']['train_zoom_range'], fallback=0)
        config['CNN']['train_zoom_range'] = self.train_zoom_range
        
        self.train_horizontal_flip = self.get_boolean(config['CNN']['train_horizontal_flip'], fallback=False)
        config['CNN']['train_horizontal_flip'] = self.train_horizontal_flip
        
        self.train_vertical_flip = self.get_boolean(config['CNN']['train_vertical_flip'], fallback=False)
        config['CNN']['train_vertical_flip'] = self.train_vertical_flip
        
        self.train_save_augmentation = self.get_boolean(config['CNN']['train_save_augmentation'], fallback=False)
        config['CNN']['train_save_augmentation'] = self.train_save_augmentation
        
        self.cnn_auto_save = self.get_boolean(self.config['CNN']['auto_save'] , fallback=False)
        self.config['CNN']['auto_save']  = self.cnn_auto_save
        
        self.trigger_type = config['VISION']['trigger_type']
        self.camera_name = config['CAMERA']['name']
        self.platform_model = config['PLATFORM']['model']
        self.cuda_visible_devices = config["CUDA"]["cuda_visible_devices"]

        self.mode_name = config['MODE']['name']
        
        config['MODE']['spacebar_save_roi'] = self.get_boolean(config['MODE']['spacebar_save_roi'], fallback=True)
        
        self.config['MODE']['show_live'] = self.get_boolean(self.config['MODE']['show_live'], fallback=True)
        
        self.non_stop = self.get_boolean(config['MODE']['non_stop'], fallback=False)
        config['MODE']['non_stop'] = self.non_stop
        
        self.show_running = self.get_boolean(self.config['MODE']["show_running"], fallback=True)
        self.config['MODE']["show_running"] = self.show_running
        
        self.config['MODE']['show_results'] = self.get_boolean(self.config['MODE']['show_results'], fallback=True)
        
        self.config['MODE']['always_show_on_bottom'] = self.get_boolean(self.config['MODE']['always_show_on_bottom'], fallback=False)
        
        self.queue_frames = self.get_boolean(self.config['MODE']['queue_frames'], fallback=False)
        self.config['MODE']['queue_frames'] = self.queue_frames
        
        self.images_path = Path(config['LOG']['images_path'])
        self.results_path = Path(config['LOG']['results_path'])
        self.mes_path = Path(config['LOG']['mes_path'])
        self.mes_connect_path = Path(config['CONNECT']['mes_path'])
        self.mes_connect_type = config['CONNECT']['mes_type']
        self.log_type = config['LOG']['type']
        self.reset_log = self.get_boolean(config['LOG']['reset_log'],fallback= False)
        config['LOG']['reset_log'] = self.reset_log
        
        self.alarm_if_fail_3x = self.get_boolean(self.config['MODE']['alarm_if_fail_3x'],fallback= False)
        self.config['MODE']['alarm_if_fail_3x'] = self.alarm_if_fail_3x
        
        self.tester_id = config['LOG']['tester_id']
        self.backup_path = Path(config['LOG']['backup_path'])
        
        self.debug_level = self.get_int(config['DEBUG']['level'],fallback= 40)
        config['DEBUG']['level'] = self.debug_level
        
        Logging.set_log_level(self.debug_level)
        
        Logging.info(config)
        # initialize total counts    
        self.class_total = {"PASS":0, "FAIL":0}
        try:
            with open(os.path.join("Profile",self.profile,'class_total.json')) as json_file:
                self.class_total = json.load(json_file)
        except:
            traceback.print_exc()
            os.makedirs(os.path.join("Profile",self.profile), exist_ok=True) 
            with open(os.path.join("Profile",self.profile,'class_total.json'), 'w') as fp:
                json.dump(self.class_total, fp, sort_keys=True, indent=4)
                
    # write total counts
    def write_total(self):
        with open(os.path.join("Profile",self.profile,'class_total.json'), 'w') as fp:
            json.dump(self.class_total, fp, sort_keys=True, indent=4)
            
    def save(self):
        #if self.start.Users.login_admin():
        super(FMCV_Config,self).save()
        self.write_total()
        
def init(in_start):
    global start
    start = in_start