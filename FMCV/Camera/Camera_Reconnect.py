import cv2
import time
print("Loading Camera Python Script")

def init_camera():
    global cm1
    #cm1 = cv2.VideoCapture(1,cv2.CAP_DSHOW)
    cm1 = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    cm1.set(cv2.CAP_PROP_FRAME_WIDTH,2592)
    cm1.set(cv2.CAP_PROP_FRAME_HEIGHT,1944)
    # cm2 = cv2.VideoCapture(1)
    # cm2.set(cv2.CAP_PROP_FRAME_WIDTH,2595)
    # cm2.set(cv2.CAP_PROP_FRAME_HEIGHT,1944)
    #cm2 = cv2.VideoCapture(1,cv2.CAP_DSHOW)
    # cm1.set(cv2.CAP_PROP_FRAME_WIDTH,3264)
    # cm1.set(cv2.CAP_PROP_FRAME_HEIGHT,2448)
    #cm1.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    #cm1.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    #cm1.set(cv2.CAP_PROP_AUTOFOCUS,0)
    #cm1.set(cv2.CAP_PROP_FOCUS,450)

    ret, frm1 = cm1.read()
    #ret, frm2 = cm2.read()

def close_cap():
    global cm1
    if cm1 is not None:
        if cm1.isOpened():
            print("closing camera")
            cm1.release()
            


def get_image(name=None):
    global cm1
    ret, frm1 = cm1.read()
    #ret, frm2 = cm2.read()
    #frm2 = cv2.flip(frm2, 0)
    #return frm1
    # reconnect videocapture if error from videocapture.read()
    if not ret:  
        print("Cannot capture frame device,Retrying...")        
        close_cap()
        time.sleep(3)
        init_camera()
        
    frames = {"0":frm1}
    #frames = {"0":frm1,"1":frm2}
    #frames = {"0":frm1,"2":cv2.flip(frm1, 0),"3":cv2.flip(frm1, 1)}
    if name is not None:
       return frames[name]
    return frames
    
init_camera()