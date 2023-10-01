from pyModbusTCP.client import ModbusClient
from pyModbusTCP import utils
import threading
import traceback
import os
import time
from datetime import datetime,timedelta
from FMCV import Logging
from queue import Queue

class ModbusTCP():
    def __init__(self,start, callback):
        self.start = start
        self.callback = callback
        self.modbus_tcp = self.create_thread()  
        self.com = self.modbus_tcp.com        
        
    def create_thread(self):
        config = self.start.Config.config
        if config['CONTROL']['modbus_type'] == 'JAKA':
            com = ModbusTCPThreadJAKA(parent=self,host=config['CONTROL']['modbus_ip'],port=int(config['CONTROL']['modbus_port']))
        
        if config['CONTROL']['modbus_type'] == 'DOBOT':
            com = ModbusTCPThreadDOBOT(parent=self,host=config['CONTROL']['modbus_ip'],port=int(config['CONTROL']['modbus_port']))
                
        com.daemon=True
        
        if config['CONTROL']['modbus_ip'] != "NONE":
            com.start()
        return com
        
    def on_callback(self, thread, data):
        #print (thread, data)
        self.callback(thread,data)



class ModbusTCPThreadJAKA(threading.Thread):
    '''
        Type    Cabinate 2.1    MiniCab         Action
        Single  DI17 40         DI8  40         Robot Next Position
        Single  DI19 42         DI10 42         Vision Failed
        Single  DI20 43         DI11 43         Done sucessfully update integer and float value
        Single  DI21 44         DI12 44         MES PASS 
        
        Integer AI3 100         AI1 100    
        
        Integer AO3 96          AO1  96         Step Control
    '''

    def __init__(self, parent=None, host="10.5.5.100", port=502):
        super(ModbusTCPThreadJAKA, self).__init__()
        self.parent = parent
        if host != "NONE":
            self.com = ModbusClient(host=host, port=port, unit_id=1, auto_open=True)
        self.flag_next = False
        self.flag_stop = False
        self.flag_fail = False
        self.flag_fiducial = False
        self.flag_xyz = False
        self.flag_reset = False
        self.flag_mes_pass = False
        self.flag_mes_fail = False
        
    def close(self):
        try:
            self.com.close()
        except:
            traceback.print_exc()
            
    def run(self):
        wait_time = 0.025
        self.last_state = False
        self.last_reset_state = False
        while True: 
            try:
                regs = self.com.read_discrete_inputs(9, 1) #DO18
                if regs:
                    if regs[0] != self.last_reset_state:
                        self.last_reset_state = regs[0]
                        if regs[0]:
                            Logging.info("Robot DO18(DO8) reset ",regs[0])
                            self.parent.on_callback(self, "RESET")
                            self.reset()
                else:
                    Logging.info("JAKA modbus read error")
                    
                step = self.com.read_input_registers(96, 1)[0] #Su AO3 Minicobot AO1
                #print(step[0])

                regs = self.com.read_discrete_inputs(8, 1) #DO17
                if regs:
                    if regs[0] != self.last_state:
                        self.last_state = regs[0]
                        Logging.info("Trigger vision DO17(DO8) is ",regs[0])
                        if regs[0]:
                            if step > 0 :
                                #step = step - 1
                                Logging.info(f"Trigged with step : {step} type:{type(step)}")
                                self.parent.on_callback(self, f"T,{step}")
                            else:
                                Logging.info("Trigged")
                                self.parent.on_callback(self, "T")
                        
                        # Clear to cobot signal
                        if self.last_state == False:
                            self.reset()

                else:
                    Logging.info("JAKA modbus read error")

                if self.flag_mes_pass:
                    self.com.write_single_coil(44,1) #DI21(DI12)
                    self.flag_mes_pass = False
                    
                if self.flag_mes_fail:
                    self.com.write_single_coil(44,0) #DI21(DI12)
                    self.flag_mes_fail = False

                if self.flag_fail:
                    #time.sleep(wait_time)
                    self.com.write_single_coil(42,1) #DI19(DI11)
                    self.flag_fail = False
                    
                if self.flag_next:    
                    #time.sleep(wait_time)
                    self.com.write_single_coil(40,1) #DI17(DI8)
                    self.flag_next = False

                if self.flag_stop:
                    #time.sleep(wait_time)
                    self.com.write_single_coil(40,0) #DI17(DI8)
                    self.flag_stop = False
                    
                if self.flag_fiducial:
                    self.flag_fiducial = False
                    x, y, x_mm, y_mm, rz = self.fiducial
                    self.com.write_single_register(100, 0) #AI3
                    self.com.write_single_register(101, 0) #AI4
                    self.com.write_single_register(102, 0) #AI5
                    self.com.write_single_register(103, 0) #AI6
                    
                    if x>=0:
                        self.com.write_single_register(100, x) #AI3
                    else:
                        self.com.write_single_register(101, abs(x)) #AI4
                    if y>=0:
                        self.com.write_single_register(102, y) #AI5
                    else:
                        self.com.write_single_register(103, abs(y)) #AI6
                        
                    Logging.info("update jaka offset x = {} y = {} and mm x = {} mm y = {}".format(x, y, x_mm, y_mm))

                    self.com.write_multiple_registers(132, self.get_ieee_modbus_float(x_mm)) #AI35
                    self.com.write_multiple_registers(134, self.get_ieee_modbus_float(y_mm)) #AI36
                    self.com.write_multiple_registers(146, self.get_ieee_modbus_float(rz)) #AI42
                    
                    self.com.write_single_coil(43,1) #DI20(DI11) Done Integer Float Value Update
                    
                if self.flag_reset:
                    self.flag_reset = False
                    Logging.info("Clear signal called")
                    self.reset()
                    
                if self.flag_xyz:
                    self.flag_xyz = False
                    x, y, z, rx, ry, rz = self.xyz
                    
                    Logging.info("Update jaka 3D x = {} y = {} z = {}\n  rx = {} ry = {} rz = {}".format(x, y, z, rx, ry, rz))
                    
                    self.com.write_multiple_registers(136, self.get_ieee_modbus_float(x)) #AI37
                    self.com.write_multiple_registers(138, self.get_ieee_modbus_float(y)) #AI38
                    self.com.write_multiple_registers(140, self.get_ieee_modbus_float(z)) #AI39
                    self.com.write_multiple_registers(142, self.get_ieee_modbus_float(rx)) #AI40
                    self.com.write_multiple_registers(144, self.get_ieee_modbus_float(ry)) #AI41
                    self.com.write_multiple_registers(146, self.get_ieee_modbus_float(rz)) #AI42
                    
                    self.com.write_single_coil(43,1) #DI20 Done Integer Float Value Update
                    
            except:
                Logging.info("Modbus Thread Run Exception")
                traceback.print_exc() 
                time.sleep(1)
                self.flag_reset = True
            #time.sleep(wait_time)
            
    def get_ieee_modbus_float(self,val):
        val_encode = utils.encode_ieee(val, double=False)
        val_encoded = utils.long_list_to_word([val_encode], big_endian=True, long_long=False)
        return val_encoded
        

    def reset(self):
        Logging.info("Turn Off DI17(DI8)[ACK] DI19(DI10)[NG] DI20(DI11) DI21(DI12)[Value OK]")
        self.com.write_single_coil(44,0) #DI21  MES
        self.com.write_single_coil(43,0) #DI20  OK with VALUE
        self.com.write_single_coil(42,0) #DI19  FAILED
        self.com.write_single_coil(40,0) #DI17  NEXT
        
    def fiducial_offset(self, x, y, x_mm, y_mm, angle):
        self.flag_fiducial = True
        self.fiducial = [x, y, x_mm, y_mm, angle] # angle write to RZ
    
    def update_xyz(self, x, y, z, rx, ry, rz):
        self.flag_xyz = True
        self.xyz = [x, y, z, rx, ry, rz]
    
    def go_next(self):
        self.flag_next = True

    def stop(self):   
        self.flag_stop = True
    
    def fail(self):
        self.flag_fail = True
        
    def mes_pass(self):
        self.flag_mes_pass = True

    def mes_fail(self):
        self.flag_mes_fail = True
            
class ModbusTCPThreadDOBOT(threading.Thread):
    def __init__(self, parent=None, host="192.168.1.6", port=502):
        super(ModbusTCPThreadDOBOT, self).__init__()
        self.parent = parent
        if host != "NONE":
            self.com = ModbusClient(host=host, port=port, unit_id=1, auto_open=True)
    
    def close(self):
        try:
            self.com.close()
        except:
            traceback.print_exc()
            
    def run(self):        
        while True: 
            try:             
                regs = self.com.read_holding_registers(10, 1)
                if regs:
                    #print(regs[0])
                    if regs[0] == 1:
                        self.com.write_single_register(10,0)
                        print("Dobot RESET")
                        self.parent.on_callback(self, "RESET")
                        
                else:
                    print("read error")
                
                regs = self.com.read_holding_registers(1, 1)
                if regs:
                    #print(regs[0])
                    if regs[0] == 1:
                        self.com.write_single_register(1,0)
                        print("trigged")
                        self.parent.on_callback(self, "T")                       
                else:
                    print("read error")
                    
            except:
                print("Modbus Thread Run Exception")
                traceback.print_exc() 
                time.sleep(1)
            time.sleep(0.1)
            
    def go_next(self):
        try:
            self.com.write_single_register(1,0)
            self.com.write_single_register(2,1)
            regs = self.com.read_holding_registers(1, 1)
            print(regs[0])
            regs = self.com.read_holding_registers(2, 1)
            print(regs[0])
            print("Modbus Dobot Go Next")
        except:
            print("Modbus Thread Run Exception")
            traceback.print_exc()
            
    def stop(self):
        pass
        
    def fail(self):
        self.flag_fail = True
        print("DOBOT fail output")
            
if __name__ == '__main__' :
    # import time
    # def callback(t,c):
        # print(f'{t},{c}')
    # manager(callback)
    # while True:
        # time.sleep(0.1)
    pass