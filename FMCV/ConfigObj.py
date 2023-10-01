import os
import traceback
import json
from configparser import ConfigParser

from pathlib import Path
from datetime import datetime

from FMCV import ObjDict
from FMCV import Logging


_UNSET = object()

class ConfigObj():
    '''
        Manager of single config file with automatic backup
        override get_config_ready() for ready the variable from string to object type example float boolean
    '''
    def __init__(self, default_config : ConfigParser, CONFIG_PATH, BACKUP_PATH, CONFIG_FILENAME):
        
        # Logic Config Path
        self.CONFIG_PATH = Path(CONFIG_PATH)
        self.BACKUP_PATH = Path(CONFIG_PATH,BACKUP_PATH)
        self.CONFIG_FILE = Path(CONFIG_PATH,CONFIG_FILENAME)
        self.CONFIG_FILENAME = CONFIG_FILENAME
        
        # Configparser Type
        self.default_config = default_config
        self.config = ObjDict.config2obj(default_config)
            
        # Read config file
        # self.read_config()
    
    def new_config():
        return ConfigParser()
    
    #String printing output
    def __repr__(self):
        return json.dumps(self.config, indent = 4)
    
    #String printing output
    def __str__(self):
        return json.dumps(self.config, indent = 4)  
        
    def read(self):
        try:
            #read config from CONFIG_FILE
            temp_config = ConfigParser()

            temp_config.read(self.CONFIG_FILE) # CONFIG_FILE = CONFIG_PATH + CONFIG_FILENAME
            
            # Check if need write upgraded config
            self.is_upgraded = False
            
            # upgrade default_config into new one
            for section in self.default_config.sections():
                # check if temp_config has section, add if not
                if not temp_config.has_section(section):
                    temp_config.add_section(section)
            
                # check if all self.default_config option exist in config.ini   
                if not all(item in temp_config.options(section)  for item in self.default_config.options(section)):
                    self.is_upgraded = True
                    
                # merge existing config
                self.config[section].update(temp_config[section])
             
            # Recovery existing config file section to self.default_config  
            for section in temp_config.sections():
                if not self.default_config.has_section(section):
                    self.config.update(ObjDict.dict2obj({section:ObjDict.dict2obj({})}))
                    self.config[section].update(temp_config[section])
                    Logging.debug(f"added section {section} into config")

            # get ready configparser string type to float boolean and etc for program reading
            self.get_config_ready() # check self.config if content type valid
            
            if self.is_upgraded == True:
                Logging.debug("Config is_upgraded")
                self.save()
            
        except:
            Logging.info(traceback.format_exc())
            self.write_config(self.default_config)
            
            self.config = ObjDict.config2obj(self.default_config)
            
            self.get_config_ready()

    def save(self):
        '''
        save_config does write current Object Dict Config to File
        '''
        temp_config = ObjDict.obj2config(self.config) 
        
        self.write_config(temp_config)
        
        Logging.info(self.config)
        
        print("Config saved")
        
    # Possible boolean values in the configuration.
    # Copy from configparser 3.11 https://github.com/python/cpython/blob/bbac9a8bcc4d7c0692e8b4f62a955f7ca107b496/Lib/configparser.py
    BOOLEAN_STATES = {'1': True, 'yes': True, 'true': True, 'on': True,
                      '0': False, 'no': False, 'false': False, 'off': False,
                      'y': True, 'n': False}
                      
    def _get_conv(self, value, conv, fallback=_UNSET):
        try:
            return conv(value)
        except:            
            Logging.debug(traceback.format_exc())
            Logging.debug(f'Error in file: {self.CONFIG_FILE}')
            if fallback is _UNSET:
                raise
            self.is_upgraded = True
            return fallback
            
    def _convert_to_boolean(self, value):
        """Return a boolean value translating from other types if necessary.
        """
        if value.casefold() not in self.BOOLEAN_STATES:
            raise ValueError('Not a boolean: %s' % value)
        return self.BOOLEAN_STATES[value.casefold()]
        
    def _split_comma(self,value):
        return value.split(",")

    def get_int(self, value, fallback=_UNSET):
        return self._get_conv(value, int, fallback)
            
    def get_float(self, value, fallback=_UNSET):
        return self._get_conv(value, float, fallback)
            
    def get_boolean(self, value, fallback=_UNSET):
        return self._get_conv(value,self._convert_to_boolean, fallback)
    
    def get_str_list(self, value, fallback=_UNSET):
        return self._get_conv(value, self._split_comma, fallback)
        
    def get_config_ready(self, config : ConfigParser):
        self.config
        #global config
        #global string_config
        
        #self.config = Dict2Obj.config2obj(config)

        #config.FMCV.loglevel = configparser_config.getint('FMCV','loglevel')
        #config.FMCV.profile = re.sub(r'[^-_a-zA-Z0-9 ]','', configparser_config['FMCV']['profile'])
        
        #start.log.info(f"config : {config}")
        #start.log.info(f"loglevel : {config['FMCV']['loglevel']}")
        #start.log.info(f"profile : {config['FMCV']['profile']}")
        
        #config = Dict2Obj.dict2obj(config)
        
        #test_config = configparser.ConfigParser()

        # test_config.add_section('FMCV2')
        # test_config['FMCV2']['program'] = 'landing' #uvc, image, url, hik, basler,  
        # test_config['FMCV2']['profile'] = 'default'
        # test_config['FMCV2']['loglevel'] = '0'
        
        # test_config = Dict2Obj.config2dict(test_config)
        # test_config = Dict2Obj.dict2obj(test_config)
        
        #config = Dict2Obj.merge(test_config,config)
        #config = Dict2Obj.merge(config,{"test":"123"})
        #config = Dict2Obj.merges(test_config,config,{"test":"123"})
        #print(config)
        #print(config.test)
        #pass
 
    def write_config(self, config : ConfigParser):
        try:    
            self.CONFIG_PATH.mkdir(exist_ok=True, parents=True)
            
            # Make a backup if applicable
            if Path(self.CONFIG_FILE).is_file():
                try:
                    self.BACKUP_PATH.mkdir(exist_ok=True, parents=True)     
                    
                    date_time_string = datetime.now().strftime("%Y%m%d_%H%M%S_")
                    
                    backup_file = os.path.join(self.BACKUP_PATH, date_time_string + self.CONFIG_FILENAME)
                    
                    os.replace(self.CONFIG_FILE, backup_file)
                    
                    Logging.info(f'Backup {self.CONFIG_FILE} to {backup_file}')
                except:
                    Logging.debug(traceback.format_exc())
                    Logging.info(f"No config for backup while writing {self.CONFIG_FILENAME}")
                    
            # Write logic_config.txt 
            with open(self.CONFIG_FILE, 'w') as file:
            
                Logging.info(f'Writing {self.CONFIG_FILE}')
                
                config.write(file)
                
                return True
                
        except:
            Logging.debug(traceback.format_exc())
            
        return False