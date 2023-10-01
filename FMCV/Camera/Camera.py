import cv2
import traceback
import time

print("Loading Camera Python Script")

# Constants
CAMERA_INDEX = 0  # Index of the camera (0 for the default camera)
RETRY_INTERVAL = 1  # Time (in seconds) to wait before trying to reconnect


def reconnect_camera():
    global cm1
    while not cm1.isOpened():
        cm1.release()
        print("Unable to connect to camera. Retrying in", RETRY_INTERVAL, "seconds...")
        time.sleep(RETRY_INTERVAL)
        cm1 = cv2.VideoCapture(CAMERA_INDEX)
    print("Camera reconnected!")

def open_camera():
    global cm1

    print("Connecting camera")
    cm1 = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    #cm1.set(cv2.CAP_PROP_FRAME_WIDTH,2592)
    #cm1.set(cv2.CAP_PROP_FRAME_HEIGHT,1944)

    cm1.set(cv2.CAP_PROP_AUTOFOCUS,0)
    cm1.set(cv2.CAP_PROP_FOCUS,200)

# Initialize the video capture
open_camera()

ret, frm1 = cm1.read()
#ret, frm2 = cm2.read()
def get_image(name=None):
    global cm1
    
    ret, frm1 = cm1.read()
    if not ret:
        print("Camera disconnected!")
        cm1.release()  # Close the current connection
        reconnect_camera()
        
    #ret, frm2 = cm2.read()
    #frm2 = cv2.flip(frm2, 0)
    #return frm1
    frames = {"0":frm1}
    #frames = {"0":frm1,"1":frm2}
    #frames = {"0":frm1,"2":cv2.flip(frm1, 0),"3":cv2.flip(frm1, 1)}
    if name is not None:
       return frames[name]
    return frames