from FMCV.Comm import SerialCom, ModbusTCPCom, MESCom, TCP_JAKA

try:
    from FMCV.Platform.Jaka import jkrc
except:
    print("JAKA jkrc cannot import")

import time
import traceback

from FMCV import Logging

import math

serial = None
modbusTCP = None
mes = None

def callback(obj, msg):
    global start
    
    if start.config.CONTROL.trigger_delay_seconds > 0.0:
        Logging.info(f"Trigger Delay {start.config.CONTROL.trigger_delay_seconds} seconds")
        time.sleep(start.config.CONTROL.trigger_delay_seconds)
    
    msg = ''.join(filter(str.isprintable, msg)) 
    print(f"putting message to main queue ({msg})")
    try:     
        frames = None
        if start.Config.queue_frames:
            if msg in ("T"): #or msg.startswith('T,'): To-Do
                start_time = time.time()
                frames = start.Cam.get_image()
                print(" images time ",time.time()-start_time)
                start.Com.go_next()            
                
        start.Main.main_queue.put_nowait((obj, msg, frames))
    except:
        traceback.print_exc()


def init(in_start):
    global start, modbusTCP, serial, mes
    
    global tcp_jaka
    
    start = in_start
    if start.Config.modbus_type in ("JAKA","DOBOT"):
        try:
            modbusTCP = ModbusTCPCom.ModbusTCP(start,callback)
            
        except:
            traceback.print_exc()
            
    if start.Config.config.CONTROL.tcp_type.casefold() == "jaka_tcp":
        tcp_jaka = TCP_JAKA.TCP(start,callback)
        
    if start.Config.config.CONTROL.tcp_type.casefold() == "jaka":
        # Initialize the connection with the robot
        tcp_jaka = jkrc.RC(start.config.CONTROL.tcp_ip)
        tcp_jaka.login()
        ret = tcp_jaka.get_sdk_version()
        print("SDK version is:", ret[1])
        #ret = tcp_jaka.get_controller_ip()
        #print("Available robot ip:", ret[1])
        
        start.sub("com/robot/to_tcp", to_tcp)
        start.sub("com/robot/get_tcp", get_tcp)
       
    if start.Config.comport != "NONE":
        try:
            serial = SerialCom.Serial(start,callback)
            time.sleep(2)
            serial.write_lighting()
        except:
            traceback.print_exc()
    
    if start.Config.mes_connect_type != 'NONE':
        try:
            mes = MESCom.MES(start,callback)
        except:
            traceback.print_exc()

#General section
def go_next():
    global start, modbusTCP, serial
    
    if start.Config.comport != "NONE":        
        serial.write_result(bytes(f'1\n', 'utf8'))
        print("Serial Next")
    if start.Config.modbus_type in ("JAKA","DOBOT"):
        modbusTCP.modbus_tcp.go_next()   
        print("Modbus Next")    
    
def failed():
    global start, modbusTCP, serial
    
    if start.Config.comport != "NONE":        
        serial.write_result(bytes(f'0\n', 'utf8'))
        print("Serial Fail Sent")
    if start.Config.modbus_type in ("JAKA","DOBOT"):
        modbusTCP.modbus_tcp.fail()   
        print("Modbus Fail Sent")

def stop(): 
    global start, modbusTCP, serial
    
    if start.Config.comport != "NONE":        
        serial.write_result(bytes(f'0\n', 'utf8'))
        print("Serial Stop Sent")
    if start.Config.modbus_type in ("JAKA"):
        modbusTCP.modbus_tcp.stop()   
        print("Modbus Stop Sent")

def alarm():
    failed()

def mes_fail():
    global start, modbusTCP, serial
    
    if start.Config.modbus_type in ("JAKA"):
        modbusTCP.modbus_tcp.mes_fail()
        print("Modbus MES FAIL Sent")
        
def mes_pass():  
    global start, modbusTCP, serial
    
    if start.Config.modbus_type in ("JAKA"):
        modbusTCP.modbus_tcp.mes_pass()
        print("Modbus MES PASS Sent")


# Cobot Section
def send_tcp_command(msg):  
    if start.Config.config.CONTROL.tcp_type.casefold() == "jaka_tcp":
        ret, msg = tcp_jaka.send_command(msg)
        return ret
        
def set_user_coordinate_system(coordinate, ucs_id, name):
    if start.Config.config.CONTROL.tcp_type.casefold() == "jaka_tcp":
        ret, msg = tcp_jaka.set_user_coordinate_system(coordinate,ucs_id,name)
        return ret
    
    if start.config.CONTROL.tcp_type.casefold() == "jaka":
        tcp = coordinate
        tcp[3] = tcp[3]* math.pi/180 #degree to radius
        tcp[4] = tcp[4]* math.pi/180 #degree to radius
        tcp[5] = tcp[5]* math.pi/180 #degree to radius
        return tcp_jaka.set_user_frame_data(ucs_id, tcp, name)[0] == 0
    
def select_user_coordinate_system(ucs_id):
    if start.Config.config.CONTROL.tcp_type.casefold() == "jaka_tcp":
        timeout_start = time.time()
        ret, msg = tcp_jaka.select_user_coordinate_system(ucs_id)
        status = _wait_jaka_message('base_id',ucs_id, timeout = 10,timeout_start=timeout_start)
        Logging.debug("Wait select_user_coordinate_system for", time.time()-timeout_start, "and value is =", status)
        return ret
        
    if start.config.CONTROL.tcp_type.casefold() == "jaka":
        return tcp_jaka.set_user_frame_id(ucs_id)[0] == 0

def set_tool_coordinate_system(coordinate, tcs_id, name):
    if start.Config.config.CONTROL.tcp_type.casefold() == "jaka_tcp":
        ret, msg = tcp_jaka.set_tool_coordinate_system(coordinate,tcs_id,name)
        return ret
        
    if start.config.CONTROL.tcp_type.casefold() == "jaka":
        tcp = coordinate
        tcp[3] = tcp[3]* math.pi/180 #degree to radius
        tcp[4] = tcp[4]* math.pi/180 #degree to radius
        tcp[5] = tcp[5]* math.pi/180 #degree to radius
        return tcp_jaka.set_tool_data(tcs_id, tcp,name)[0] == 0
        
        
def select_tool_coordinate_system(tcs_id):
    if start.Config.config.CONTROL.tcp_type.casefold() == "jaka_tcp":
        timeout_start = time.time()
        ret, msg = tcp_jaka.select_tool_coordinate_system(tcs_id)
        status = _wait_jaka_message('tool_id',tcs_id, timeout = 10,timeout_start=timeout_start)
        Logging.debug("Wait select_tool_coordinate_system for", time.time()-timeout_start, "and value is =", status)
        return ret

    if start.config.CONTROL.tcp_type.casefold() == "jaka":
        return tcp_jaka.set_tool_id(tcs_id)[0] == 0

def get_tcp():
    if start.Config.config.CONTROL.tcp_type.casefold() == "jaka_tcp":
        ret, msg, t1 = tcp_jaka.get_tcp()
        if ret:
            {"jx":t1[0],"jy":t1[1],"jz":t1[2],
             "rx":t1[3],"ry":t1[4],"rz":t1[5]}
            return t1
    if start.config.CONTROL.tcp_type.casefold() == "jaka":
        ret,tcp = tcp_jaka.get_tcp_position()
        if ret == 0:
            #{"jx":tcp[0], "jy":tcp[1], "jz":tcp[2],
            # "rx":tcp[3]*180/math.pi, "ry":tcp[4]*180/math.pi, "rz":tcp[5]*180/math.pi} #radius to degree
            tcp[3] = tcp[3] * 180/math.pi #radius to degree
            tcp[4] = tcp[4] * 180/math.pi #radius to degree
            tcp[5] = tcp[5] * 180/math.pi #radius to degree
            return tcp
        else:
            Logging.error("Jaka error",ret[0])

def to_tcp(tcp, speed=20, accel=100, relative_move=True):
    tcp = tcp.copy()
    if start.Config.config.CONTROL.tcp_type.casefold() == "jaka_tcp":
        timeout_start = time.time()
        ret, msg = tcp_jaka.to_tcp(tcp, mode="moveL", speed=speed, accel=accel, relative_move=relative_move)
        status = _wait_jaka_message('reach',True, timeout = 10,timeout_start=timeout_start)
        Logging.debug("Wait robot to coordinate for",time.time()-timeout_start,"and is reached =", status)
        return ret
        
    if start.config.CONTROL.tcp_type.casefold() == "jaka":
        print(tcp)
        tcp[3] = tcp[3]* math.pi/180 #degree to radius
        tcp[4] = tcp[4]* math.pi/180 #degree to radius
        tcp[5] = tcp[5]* math.pi/180 #degree to radius
        print(tcp)
        return tcp_jaka.linear_move_extend(end_pos = tcp, move_mode = int(relative_move), is_block=True, speed=speed, acc=accel, tol=0)[0]==0

def _wait_jaka_message(key, value, timeout = 10, timeout_start = time.time()):
    # timeout variable can be omitted, if you use specific value in the while condition
    #timeout = 10   # [seconds]
    #timeout_start = time.time()
    time.sleep(0.01) # waiting get_message update from JAKA port 10000
    while time.time() < timeout_start + timeout:
        status = get_status(key)
        if status == value:
            return status
        time.sleep(0.08)
       
def get_status(key):
    #key =  "reach" # mean check if cobot reach position
    if start.Config.config.CONTROL.tcp_type.casefold() == "jaka_tcp":
        if key.casefold() == 'reach':
            return tcp_jaka.get_message('inpos')
            
        if key.casefold() == 'tool_id':
            return tcp_jaka.get_message('current_tool_id')
        
        if key.casefold() == 'base_id':
            return tcp_jaka.get_message('current_user_id')
            
    if start.config.CONTROL.tcp_type.casefold() == "jaka":
        if key.casefold() == 'reach':
            ret , state = tcp_jaka.is_in_pos()
            if ret == 0:
                return bool(state)
            else:
                return False
                
def update_offset(offset):
    global start
    Logging.info("Comm offset")
    if offset.get("x") is not None and offset.get("y") is not None:
        if start.Config.modbus_type in ("JAKA"):
            Logging.info("Update Jaka Offset")
            
            x_mm = offset.get("x_mm")
            y_mm = offset.get("y_mm")
            x = offset.get("x")
            y = offset.get("y")
            rz = offset.get("angle") # write to modbus rz
            modbusTCP.modbus_tcp.fiducial_offset(x,y,x_mm,y_mm,rz)

def send_tcp_modbus(tcp):
    Logging.info("Comm send tcp")
    x, y, z, rx, ry, rz = tcp
    if start.Config.modbus_type in ("JAKA"):
        Logging.info("Update tcp to JAKA modbus AI37 to AI42 x,y,z,rx,ry,rz")
        modbusTCP.modbus_tcp.update_xyz(x, y, z, rx, ry, rz)
        
def power_on():
    if start.Config.modbus_type in ("JAKA"):
        if start.Config.config.CONTROL.tcp_type.casefold() == "jaka":
            ret = tcp_jaka.power_on()
            ret = tcp_jaka.enable_robot()
            return ret