import serial
import threading
import traceback
import os
import time
from datetime import datetime,timedelta

class Serial(object):
    def __init__(self,Handle, callback):
        self.Handle = Handle
        self.callback = callback
        self.serial = self.create_thread()  
        self.com = self.serial.com
        self.write = self.serial.write        
        self.write_result = self.serial.write_result
        
    def write_lighting(self):        
        r = self.Handle.Config.config['LIGHTING']['red']
        g = self.Handle.Config.config['LIGHTING']['green']
        b = self.Handle.Config.config['LIGHTING']['blue']
        self.serial.write(bytes(str(f'{r},{g},{b}\n'), 'utf8'))
    
    def comport_reopen(self):
        self.serial.close()
        self.serial.join()
        del self.serial
        self.serial = self.create_thread() 
        self.com = self.serial.com
        self.write = self.serial.write
        self.write_result = self.serial.write_result
        
    def create_thread(self):
        com = SerialThread(parent=self,comport=self.Handle.Config.config['CONTROL']['comport'])
        com.daemon=True
        com.start()
        return com
        
    def on_callback(self, thread, data):
        #print (thread, data)
        self.callback(thread,data)

class SerialThread(threading.Thread):

    def __init__(self, parent=None,comport = 'COM1'):
        super(SerialThread, self).__init__()
        self.parent = parent
        print(comport)
        self.com = serial.Serial()
        self.com.port = comport
        self.com.baudrate = 19200
        #self.com.parity = 'E'
        #self.com.bytesize = 8
        #com.timeout = 1
        #self.com.setDTR(False)
        
        self.start_time = time.time()
    
    def comport_reopen(self,comport = 'COM1'):
        try:
            self.com.close()
        except:
            traceback.print_exc()
        try:
            self.com = serial.Serial()
            self.com.port = comport
            self.com.baudrate = 19200  
            #self.com.open() # will be called by run()
        except:
            traceback.print_exc()
            
    def close(self):
        try:
            self.com.close()
        except:
            traceback.print_exc()
    
    
    def write_result(self,byte_arr,start_time=None):
        try:
            self.com.write(byte_arr)
            
            output = '{0}\n'.format(chr(byte_arr[0]))             
            
            feedback_datetime = datetime.utcnow() + timedelta(hours=+8)
                                
            # path_log = os.path.join("LOGS", "{}.csv".format(feedback_datetime.date()))
            
            # with open(path_log, "a") as file_log:                
                # file_log.write("{}".format(self.total_counts+1))
                # file_log.write(",")
                # file_log.write("{}".format(self.judge))
                # file_log.write(",")
                # file_log.write("{}".format(self.triggered_delta))
                # file_log.write(",")
                # file_log.write("{}".format(feedback_datetime.isoformat(sep=',')))       
                # file_log.write(",")
                # file_log.write("{}".format((feedback_datetime-self.triggered_datetime)))  
                # file_log.write(",")                    
                # file_log.write("{},".format(chr(byte_arr[0])))
                # file_log.write("\n") 
                
            end = time.time()
            if start_time is not None:
                print("{} \noutput {}".format(end - start_time,output))
            else:
                print("{} \noutput {}".format(end - self.start_time,output))
        except:
            traceback.print_exc()
            
    def write(self,byte_arr):
        try:
            self.com.write(byte_arr)
        except:
            traceback.print_exc()
            
    def run(self):
        try:
            self.judge = 0
        
            self.total_counts = 0
            
            flag_counter = False
        
            self.com.open()
            
            time_zone = +8
            self.triggered_datetime = datetime.utcnow() + timedelta(hours=time_zone)
            self.triggered_delta = self.triggered_datetime - self.triggered_datetime
            unit_datetime = datetime.utcnow() + timedelta(hours=time_zone)
            unit_delta = self.triggered_datetime - self.triggered_datetime
            counter = 0
            
            while True:    
                #os.makedirs("LOGS", exist_ok=True)    
        
                if not flag_counter: # Activated with R LF
                    line = self.com.readline() # with a LINEFEED 0x0A
                else:                
                    line = self.com.read(16) # read 4x int
                    
                self.start_time = time.time()
                
                #print(str(line))
                
                # if flag_counter:    
                    # flag_counter = False
                    # try:
                        # total_good = line[0:4]
                        # total_good = total_good[::-1]
                        # #print(str(total_good))
                        # total_good=int.from_bytes(total_good, byteorder='big')
                        # print(str(total_good))
                        
                        # total_no_good = line[4:8]
                        # total_no_good = total_no_good[::-1]
                        # #print(str(total_no_good))
                        # total_no_good=int.from_bytes(total_no_good, byteorder='big')
                        # print(str(total_no_good))
                        
                        # self.total_counts = total_good + total_no_good
                        # print("Total = {}".format(self.total_counts))
                        
                        # total_top_ng = line[8:12]
                        # total_top_ng = total_top_ng[::-1]
                        # #print(str(total_top_ng))
                        # total_top_ng = int.from_bytes(total_top_ng, byteorder='big')
                        # print(total_top_ng)
                        
                        # total_bottom_ng = line[12:16]
                        # total_bottom_ng = total_bottom_ng[::-1]
                        # #print(str(total_bottom_ng))
                        # total_bottom_ng = int.from_bytes(total_bottom_ng, byteorder='big')
                        # print(total_bottom_ng)                   
                        
                    # except:
                        # traceback.print_exc()
                
                #elif "R" in str(line):
                #    flag_counter =  True

                # elif "T" in str(line): #0x54 0x0A
                    # current_datetime = datetime.utcnow()       
                    # current_datetime += timedelta(hours=time_zone) 
                    # self.triggered_delta = current_datetime - self.triggered_datetime
                    # self.triggered_datetime = current_datetime 
                    # #print("t {}".format(triggered_delta))
                    # self.judge+=1
                    # self.parent.on_callback(self, "T")
                    
                # elif "P" in str(line): #0x50 0x0A
                    # current_datetime = datetime.utcnow()
                    # self.judge = 0
                    # counter = counter + 1
                    # current_datetime += timedelta(hours=time_zone) 
                
                    # unit_delta = current_datetime - unit_datetime
                    # unit_datetime = current_datetime 
                                    
                    # print("Present")
                    
                    # path_log = os.path.join("LOGS", "{}.csv".format(current_datetime.date()))
                    # with open(path_log, "a") as file_log:                
                        # file_log.write("{}".format(self.total_counts+1))
                        # file_log.write(",")
                        # file_log.write("{}".format(self.judge))
                        # file_log.write(",")
                        # file_log.write("{}".format(unit_delta))
                        # file_log.write(",")
                        # file_log.write("{}".format(unit_datetime.isoformat(sep=',')))       
                        # file_log.write(",,,,P")                 
                        # file_log.write("\n") 
                    
                # elif "E" in str(line): #0x45 0x0A

                    # current_datetime = datetime.utcnow()
                    
                    # current_datetime += timedelta(hours=time_zone) 
                
                    # unit_delta = current_datetime - unit_datetime
                    # unit_datetime = current_datetime 
                                    
                    # self.parent.on_callback(self, "E")
                    # print("Ejected")
                    
                    # path_log = os.path.join("LOGS", "{}.csv".format(current_datetime.date()))
                    # with open(path_log, "a") as file_log:                
                        # file_log.write("{}".format(self.total_counts))
                        # file_log.write(",")
                        # file_log.write("{}".format(self.judge))
                        # file_log.write(",")
                        # file_log.write("{}".format(unit_delta))
                        # file_log.write(",")
                        # file_log.write("{}".format(unit_datetime.isoformat(sep=',')))       
                        # file_log.write(",,,,E")  
                        # file_log.write("\n") 
                    
                # elif "S" in str(line):
                    # os.system("shutdown /s /t 1")
                line = line.replace(b'\r', b'')
                line = line.replace(b'\n', b'')
                
                self.parent.on_callback(self,line.decode())    
                self.com.flush()         
        except:
            print("Serial Thread Run Exception")
            traceback.print_exc()    
            
if __name__ == '__main__' :
    # import time
    # def callback(t,c):
        # print(f'{t},{c}')
    # manager(callback)
    # while True:
        # time.sleep(0.1)
    pass