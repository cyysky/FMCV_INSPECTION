import logging
import os
import inspect
import queue
from threading import Thread,Lock
import traceback
import time

cam_queue = queue.Queue(maxsize=1)

images = {}

_lock = Lock()

def camera_update():
    global cam_queue, start, images, _lock
    
    while True:
        try:
            if start.ViewEvent.live:
                with _lock:
                    images = start.Camera.get_image()
                    time.sleep(0.01) # To allow main thread run
            else:
                time.sleep(0.5)
        except:
            traceback.print_exc()

def init(in_start):
    global start
    start = in_start
    trigger_thread = Thread(target=camera_update, daemon=True)
    trigger_thread.start()

def get_live():
    return images

def get_image():
    with _lock:
        return start.Camera.get_image()
        
def get_current_image():
    return list(get_image().values())[start.MainUi.cam_pos]