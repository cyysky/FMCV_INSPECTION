import serial
import threading
import traceback
import os
import time
from datetime import datetime,timedelta
import shutil

from pathlib import Path

import watchdog
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

class MES(object):
    def __init__(self, start, callback):
        self.start = start
        self.callback = callback
        
        if start.Config.mes_connect_type == "VS":
            mes = VsMesThread(parent=self)
            mes.daemon=True
            mes.start()
        
    def on_callback(self, thread, data):
        print (thread, data)
        self.callback(thread,data)

class VsMesThread(threading.Thread):
    def __init__(self, parent=None):
        super(VsMesThread, self).__init__()
        self.parent = parent
        self.barcode = ""
        self.mes_path = ""

    def run(self):
        try:
            #https://thepythoncorner.com/posts/2019-01-13-how-to-create-a-watchdog-in-python-to-look-for-filesystem-changes/
            #https://github.com/gorakhargosh/watchdog
            patterns = ["*"]
            ignore_patterns = None
            ignore_directories = False
            case_sensitive = True
            my_event_handler = PatternMatchingEventHandler(patterns, ignore_patterns, ignore_directories, case_sensitive)

            my_event_handler.on_created = self.on_created
            my_event_handler.on_deleted = self.on_deleted
            my_event_handler.on_modified = self.on_modified
            my_event_handler.on_moved = self.on_moved

            self.mes_path = path = self.parent.start.Config.mes_connect_path
            os.makedirs(path, exist_ok=True)
            my_observer = Observer()
            my_observer.schedule(my_event_handler, path, recursive=False)
            my_observer.start()
            while True:
                time.sleep(1)
        except:
            traceback.print_exc()
            
    def process_input_file(self,event):
        self.barcode = ""
        print(f"{event.src_path} has been created!")
        try:
            #https://pythonsolved.com/python-get-first-line-of-file/
            if Path(event.src_path).is_file():
                with open(event.src_path, "r") as file:
                    first_line = file.readline()
                    print(first_line)
                    line_split = first_line.split("|")
                    for text in line_split:
                        print(text)
                        if text[:4] == "CNR=":
                            print(text[4:])
                            self.barcode = text[4:]
                            self.parent.callback(self, "VS_MES")
                os.makedirs(os.path.join(self.mes_path,"ACK"), exist_ok=True)
                current_datetime = datetime.utcnow() + timedelta(hours=+8)
                shutil.move(event.src_path, os.path.join(self.mes_path,"ACK",current_datetime.strftime("%Y%m%d_%H%M%S")))
        except:
            traceback.print_exc()
            
    def on_created(self,event):
        self.process_input_file(event) # Process input file
            
    def on_deleted(self,event):
        print(f"Deleted {event.src_path}!")

    def on_modified(self,event):
        print(f"{event.src_path} has been modified")
        self.process_input_file(event) # Process input file

    def on_moved(self,event):
        print(f"Moved {event.src_path} to {event.dest_path}")
        
            
if __name__ == '__main__' :
    pass