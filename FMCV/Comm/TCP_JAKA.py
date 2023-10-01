import threading
import traceback
import os
import time
import json
import socket
from threading import Thread, Lock
from queue import Queue
from queue import Empty
from FMCV import Logging

class TCP():
    '''
        coordinate is list [x, y, z, rx, ry, rz]
    '''

    def __init__(self,start,callback):
        self.start = start
        self.callback = callback
        self.tcp = self.create_thread()
        #self.msg = self.create_message_thread()
        
    def on_callback(self, thread, data):
        self.callback(thread,data)
        
    def create_thread(self):
        config = self.start.Config.config
        if config['CONTROL']['tcp_type'].casefold() == "jaka":
            tcp = TCPJAKABackend(host=config['CONTROL']['tcp_ip'],port=10001)
            tcp.setDaemon(True)
            tcp.start()
            return tcp
        return None
        
    def create_message_thread(self):
        config = self.start.Config.config
        if config['CONTROL']['tcp_type'].casefold() == "jaka":
            msg = TCPJAKAMessageBackend(host=config['CONTROL']['tcp_ip'],port=10000)
            msg.setDaemon(True)
            msg.start()
            return msg
        return None
        
    def _unstable_get_message(self,key = None):
        time.sleep(0.01) # prevent TCPJAKAMessageBackend thread stop
        if key is not None:
            return self.msg.get_message().get(key)
        else:
            return self.msg.get_message()
            
    def get_message(self,key = None):
        if self.tcp is not None:
            self.tcp.queue.put('{"cmdName":"get_data"}')
            returned_value = self.tcp.queue_return.get(timeout=10)
            ret, msg = self._return_value("get_data",returned_value)
            if ret:
                if key is not None:
                    return returned_value.get(key)
                else:
                    return returned_value
                    
        
    def send_command(self,msg):
        if self.tcp is not None:
            self.tcp.queue.put(msg)
            return self.tcp.queue_return.get(timeout=10)
            
    def get_tcp(self):
        if self.tcp is not None:
            self.tcp.queue.put('{"cmdName":"get_tcp_pos"}')
            returned_value = self.tcp.queue_return.get(timeout=10)
            respond = self._return_value("get_tcp_pos",returned_value)
            return respond[0],respond[1], returned_value.get("tcp_pos")
            
    def to_tcp(self, coordinate, mode="moveL", speed=20, accel=100, relative_move=True):
        if self.tcp is not None:
            if mode == "end_mode":
                position = f'"endPosition":{coordinate}'
            elif mode == "moveL":
                position = f'"cartPosition":{coordinate}, "relFlag":{int(relative_move)}'
            command = f'{{"cmdName":"{mode}", {position}, "speed":{speed}, "accel":{accel}}}'
            self.tcp.queue.put(command)
            return self._return_value(mode)
    
    def power_on(self):
        if self.tcp is not None:
            self.tcp.queue.put('{"cmdName":"power_on"}')
            return self._return_value("power_on")
    
    def enable_robot(self):
        if self.tcp is not None:
            self.tcp.queue.put('{"cmdName":"enable_robot"}')
            return self._return_value("enable_robot")

    def set_user_coordinate_system(self, coordinate, ucs_id, name):
        if self.tcp is not None:
            self.tcp.queue.put(f'{{"cmdName":"set_user_offsets","userffset":{coordinate},"id":{ucs_id},"name":"{name}"}}')
            return self._return_value("set_user_offsets")
                
    def select_user_coordinate_system(self, ucs_id):
        if self.tcp is not None:
           self.tcp.queue.put(f'{{"cmdName":"set_user_id","user_frame_id":{ucs_id}}}')
           return self._return_value("set_user_id")
           
    def set_tool_coordinate_system(self, coordinate, tcs_id, name):
        if self.tcp is not None:
            self.tcp.queue.put(f'{{"cmdName":"set_tool_offsets","tooloffset":{coordinate},"id":{tcs_id},"name":"{name}"}}')
            return self._return_value("set_tool_offsets")
                
    def select_tool_coordinate_system(self, tcs_id):
        if self.tcp is not None:
           self.tcp.queue.put(f'{{"cmdName":"set_tool_id","tool_id":{tcs_id}}}')
           return self._return_value("set_tool_id")

    def _return_value(self,command,returned_value=None):
        if not returned_value:
            returned_value = self.tcp.queue_return.get(timeout=10)
            while not self.tcp.queue_return.empty():
                try:
                    self.tcp.queue_return.get_nowait()
                    self.tcp.queue_return.task_done()
                except Queue.Empty:
                    break
            
        if returned_value.get("errorCode") == "0" and command == returned_value.get("cmdName"):
            ret_val = True
        else:
            ret_val = False
            
        Logging.debug("{},{}".format(ret_val, returned_value.get("errorMsg")))
        return ret_val, returned_value.get("errorMsg")
        

class TCPJAKABackend(Thread):
    def __init__(self, host="10.5.5.100", port=10001):
        '''
            Usage
            jaka_instruct_thread = TCPJAKA(jaka_que)
            jaka_instruct_thread.setDaemon(True)
            jaka_instruct_thread.start()
            
            jaka_instruct_thread.queue.put('{"cmdName":"get_tcp_pos"}')
            jaka_instruct_thread.queue_return.get("tcp_pos")
        '''
        # constructor
        # execute the base constructor
        Thread.__init__(self)
        self.host = host
        self.port = port
        # set a default value
        self.value = {}
        self.queue = Queue()
        self.queue_return = Queue()
        self.socket = None
        
    # function executed in a new thread
    def run(self):
        self._reconnect()

        # message you send to server
        while True:
            try:
                message = self.queue.get()
                Logging.debug(message)
                
                while not self.queue.empty():
                    try:
                        self.queue.get_nowait()
                        self.queue.task_done()
                    except Queue.Empty:
                        break
                
                # message sent to server
                self.socket.send(message.encode('ascii'))

                # message received from server
                data1 = self.socket.recv(5120)
                Logging.debug(data1)
                # print the received message
                # here it would be a reverse of sent message
                try:
                    self.value = json.loads(str(data1.decode('ascii')))
                except:
                    data2 = self.socket.recv(5120)
                    Logging.debug(data2)
                    data1 += data2
                    self.value = json.loads(str(data1.decode('ascii')))
                    #self.value = self._retry(message)
                
                self.queue_return.put_nowait(self.value)
                #print('Received from the server :',self.value)
                
            #except (socket.timeout, ConnectionError):
            except:
                Logging.error(traceback.format_exc())
                self.queue_return.put_nowait({})
                self._reconnect()
            
        # close the connection
        self.socket.close()
        
    def _retry(self, message):
        for a in range(10):
            try:
                # message sent to server
                self.socket.send(message.encode('ascii'))

                # message received from server
                data = self.socket.recv(5120)
                
                return json.loads(str(data.decode('ascii')))
            except:
                Logging.error("Retry get message",a)

    def _reconnect(self):
        if self.socket is not None:
            self.socket.close()

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while True:
            try:
                self.socket.connect((self.host, self.port))
                self.socket.settimeout(10)
                break
            except (socket.timeout, ConnectionError, socket.error):
                Logging.error("Failed to reconnect 10001, retrying...")
                time.sleep(1)
        
class TCPJAKAMessageBackend(Thread):
    def __init__(self, host="10.5.5.100", port=10000):
        # constructor
        # execute the base constructor
        Thread.__init__(self)
        self.host = host
        self.port = port
        # set a default value
        self.value = {}
        self._lock = Lock()
        self.socket = None
        
    # function executed in a new thread
    def run(self):
        # connect to server on local computer
        self._reconnect()
        
        # message you send to server
        while True:
            try:
                # message received from server
                data1 = self.socket.recv(5120)
                with self._lock:
                    try:
                        self.value = {}
                        self.value = json.loads(str(data1.decode('ascii'))) 
                    except:                    
                        data2 = self.socket.recv(5120)
                        data1 += data2
                        try:
                            self.value = json.loads(str(data1.decode('ascii')))
                        except:
                            data3 = self.socket.recv(5120)
                            data1 += data3
                            try:
                                self.value = json.loads(str(data1.decode('ascii')))
                            except:
                                pass
                                Logging.debug("JAKA message thread dropped and resumed")
                                #Logging.info(data1)
                                #Logging.info(traceback.format_exc())
            except:
                Logging.error(traceback.format_exc())
                self._reconnect()
                
        # close the connection
        self.socket.close()

    def _reconnect(self):
        if self.socket is not None:
            self.socket.close()

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while True:
            try:
                self.socket.connect((self.host, self.port))
                self.socket.settimeout(10)
                break
            except (socket.timeout, ConnectionError, socket.error):
                Logging.error("Failed to reconnect 10000, retrying...")
                time.sleep(1)
        
        
    def get_message(self):
        with self._lock:
            #Logging.info(self.value)
            return self.value