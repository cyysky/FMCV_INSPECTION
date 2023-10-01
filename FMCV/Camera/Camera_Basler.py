'''
A simple Program for grabing video from basler camera and converting it to opencv img.
Tested on Basler acA1300-200uc (USB3, linux 64bit , python 3.5)
'''
#https://github.com/basler/pypylon/blob/master/samples/opencv.py
from pypylon import pylon
import cv2
import traceback
# conecting to the first available camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

# Grabing Continusely (video) with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
converter = pylon.ImageFormatConverter()

# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    
def get_image(name=None):
    try:
        grabResult = camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
        #ret, frm2 = cm2.read()
        #frm2 = cv2.flip(frm2, 0)
        #return frm1
        if grabResult.GrabSucceeded():
            image = converter.Convert(grabResult)
            img = image.GetArray()
        frames = {"0":img}
        #frames = {"0":frm1,"1":frm2}
        #frames = {"0":frm1,"2":cv2.flip(frm1, 0),"3":cv2.flip(frm1, 1)}
        if name is not None:
           return frames[name]
        return frames
    except:
        traceback.print_exc()
        print("basler retrive image failed")
        return {"0":None}

