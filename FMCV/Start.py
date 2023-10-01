import os
from pathlib import Path
import datetime
import traceback
from FMCV import Version

BRAND = Version.BRAND 
version = Version.version 

#https://stackoverflow.com/questions/1676835/how-to-get-a-reference-to-a-module-inside-the-module-itself
import sys
start = sys.modules[__name__]

#Command line control
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-p","--profile", help="Specific A profile to start with", default = None)
args = parser.parse_args()

from FMCV import Logging
Logging.init(start)
log = Logging

#Start Channel for publish subscribe
from FMCV import Channel
Channel.init(start)
pub = Channel.pub
sub = Channel.sub

#Start FMCV Module
from FMCV import Util
from FMCV.Cv import Cv, Match
from FMCV.Automata import Calibrate2d
Calibrate2d.init(start)
from FMCV.Automata import Calibrate3d
Calibrate3d.init(start)

from FMCV.Config import FMCV_Config
#Read Basic Configuration
Config = FMCV_Config(start)
Config.read()
config = Config.config

from FMCV import Users
Users.init(start)

from FMCV import Profile
Profile.init(start)

from FMCV import Process
Process.init(start)

os.environ["CUDA_VISIBLE_DEVICES"] = Config.config["CUDA"]["cuda_visible_devices"]

from FMCV.Ai import CNN
CNN.init(start)

from FMCV.Ai import ANO
ANO.init(start)

from FMCV.Ui import ActionUi
ActionUi.init(start)

from FMCV import Thread
Thread.init(start)

from FMCV.Host import Host
host = Host(start, None)

from FMCV import Log
Log.init(start)

from FMCV.Ui import MainUi
MainUi.init(start)
# Some legacy code here, some later added module already using object base tkinter widgets
from FMCV.Ui import MainUi_Aux 
from FMCV.Ui import MainUi_Edit 
from FMCV.Ui import MainUi_Refresh 
from FMCV.Ui import MainUi_Results 

from FMCV.Ui import ViewEvent
ViewEvent.init(start)
    
from FMCV.Ui import RoiBusket
RoiBusket.init(start)

from FMCV.Ui import OperationResults
OperationResults.init(start)

from FMCV.Ui import RoiSearch
RoiSearch.init(start)

from FMCV import Main
Main.init(start)
Main.reset()

Camera = None
exec("from FMCV.Camera import {} as Camera".format(Config.config['CAMERA']['name']))

from FMCV import Cam
Cam.init(start)

Serial = None
ModbusTCP = None

from FMCV import Com
Com.init(start)

from FMCV.Platform.Jaka import JakaControl
JakaControl.init(start)

from FMCV.Platform.Platform import Platform, CartesianPosition
MovingPlatform = None
Main.init_moving_platform()

Channel.list_topics()

def run():
    print("Ready =========================================================================")
    MainUi.view.after(300, ViewEvent.update_view)
    MainUi.top.mainloop()