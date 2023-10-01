import threading
import traceback
import os
import time
import json
import socket
from threading import Thread
from queue import Queue, Empty

from FMCV import Logging

class Host():
    def __init__(self,start,callback):
        self.start = start
        self.callback = callback
        self.tcp_dialogue = None
        self.tcp_broadcast = None
        self.create_thread()
        
    def on_callback(self, thread, data):
        self.callback(thread,data)
        
    def create_thread(self):
        config = self.start.Config.config
        if config.HOST.dialog_port > 1024 and config.HOST.dialog_port < 65535:
            self.tcp_dialogue = TcpHostDialogue(self)
            self.tcp_dialogue.setDaemon(True)
            self.tcp_dialogue.start()
            
        if config.HOST.broadcast_port > 1024 and config.HOST.broadcast_port < 65535:
            self.tcp_broadcast = TcpHostBroadcast(self)
            self.tcp_broadcast.setDaemon(True)
            self.tcp_broadcast.start()
            
    def broadcast_message(self,msg):
        if self.tcp_broadcast is not None:
            Logging.info(msg)
            self.tcp_broadcast.message_queue.put(msg)
            
    def respond(self,msg):
        if self.tcp_dialogue is not None:
            self.tcp_dialogue.queue.put(msg)


class TcpHostDialogue(Thread):
    def __init__(self,parent, host="127.0.0.1", port=10922):
        # constructor
        # execute the base constructor
        Thread.__init__(self)
        self.parent = parent
        
        self.host = host
        self.port = port
        
        # set a default value
        self.value = {}
        self.queue = Queue()
        self.queue_return = Queue()
        
        
    # function executed in a new thread
    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            Logging.info("Dialogue host ",socket.gethostname()," Hosted on port ", self.port)
            s.bind(('', self.port))
            s.listen()
            
            while True:
                conn, addr = s.accept()     # Establish connection with client.
                while True:
                    try:
                        data = conn.recv(5120)
                        Logging.info(f"{addr} ' >> ' {data}")
                        if not data:
                            Logging.info(f'{addr} disconnected')
                            break
                        else:
                            self.parent.on_callback(self,data.decode('ascii'))
                            a = "NA"
                            try:
                                a = self.queue.get(timeout=10)
                            except:
                                Logging.info("No valid command/respond 10 timeout")
                            conn.sendall(a.encode('ascii'))
                    except:
                        traceback.print_exc()
                        break   
                        
class TcpHostBroadcast(Thread):
    def __init__(self,parent, host="127.0.0.1", port=10923):
        # constructor
        # execute the base constructor
        Thread.__init__(self)
        self.parent = parent
        
        self.host = host
        self.port = port
        
        # set a default value
        self.value = {}
        
        # create a queue to store messages to be broadcasted
        self.message_queue = Queue()
        
        # for incoming
        self.queue_return = Queue()
        
        # create a list to store connected clients
        self.client_list = []
        

    def client_thread(self,client_socket, address):
        while True:
            # receive data from the client
            data = client_socket.recv(5120)

            if not data:
                # if the data is empty, the client has disconnected
                self.client_list.remove(client_socket)
                Logging.info(f'Connection from {address} has been closed!')
                break
            else:
                # otherwise, add the message to the message queue
                self.message_queue.put(str(data.decode('ascii')))

        # close the client socket
        client_socket.close()
        
    def broadcast_thread(self):
        while True:
            # get the next message from the message queue
            message = self.message_queue.get()

            # send the message to all connected clients
            for client in self.client_list:
                client.sendall(message.encode('ascii'))

            # mark the message as processed
            #message_queue.task_done()        
        
    # function executed in a new thread
    def run(self):
        # start the broadcast thread
        broadcast_thread = threading.Thread(target=self.broadcast_thread)
        broadcast_thread.daemon = True
        broadcast_thread.start()
    
        
        # create a socket object
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            Logging.info("Brocast host ",socket.gethostname()," Hosted on port ", self.port)
            # bind the socket to a specific IP address and port
            server_socket.bind(("", self.port))
            # listen for incoming connections
            server_socket.listen()
            while True:
                # accept a new connection
                client_socket, address = server_socket.accept()
                Logging.info(f'Connection from {address} has been established!')

                # add the new client to the list
                self.client_list.append(client_socket)

                # start a new thread to handle the client
                temp_client_thread = threading.Thread(target=self.client_thread, args=(client_socket, address))
                temp_client_thread.daemon = True
                temp_client_thread.start()