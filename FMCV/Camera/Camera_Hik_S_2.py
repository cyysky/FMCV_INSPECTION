print("Loading HikRobot Camera Single Trigger Python Script")

# -- coding: utf-8 --
import os
import sys
import time
import copy
import ctypes
import traceback

from ctypes import *
import cv2
import numpy as np

from tkinter import messagebox as mb

sys.path.append("C:\\Program Files (x86)\\MVS\\Development\\Samples\\Python\\MvImport")
os.add_dll_directory("C:\\Program Files (x86)\\Common Files\\MVS\\Runtime\\Win64_x64")
from MvCameraControl_class import *

from PIL import Image as pil_image


imageSize = (-1,-1) #unused

nConnectionNum = 1 #input("Please input the number of the device to connect:")

cam = MvCamera()

def stopCamera():
    global cam
    ret = 1
    loop = 0 
    # ch:停止取流 | en:Stop grab image
    #while ret != 0:
    ret = cam.MV_CC_StopGrabbing()
    if ret != 0:
        print ("Stop Grabbing fail! ret[0x%x]" % ret)
        loop += 1
        #sys.exit()
    ret = 1
    loop = 0
    # ch:关闭设备 | Close device
    #while ret != 0:
    ret = cam.MV_CC_CloseDevice()
    if ret != 0:
        print ("Close Deivce fail! ret[0x%x]" % ret)
        loop += 1
        #sys.exit()
    ret = 1
    loop = 0
    # ch:销毁句柄 | Destroy handle
    #while ret != 0:
    ret = cam.MV_CC_DestroyHandle()
    if ret != 0:
        print ("Destroy Handle fail! ret[0x%x]" % ret)
        loop += 1
            #sys.exit()

def getCameraList():
    global cam

    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE

    # ch:枚举设备 | en:Enum device
    ret = cam.MV_CC_EnumDevices(tlayerType, deviceList)
    if ret != 0:
        print("Enum Devices fail! ret[0x%x]" % ret)
        #sys.exit()

    if deviceList.nDeviceNum == 0:
        print("Find No Devices!")
        #sys.exit()

    for i in range(0, deviceList.nDeviceNum):
        mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
            print("Gige Device: [%d]" % i)
            print("User Defined Name: ")
            for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chUserDefinedName:
                print(chr(per))
            print("%s" % "")
            print("Ip: %x\n" % mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp)
        if mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
            print("U3V Device: [%d]" % i)
            print("User Defined Name: ")
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chUserDefinedName:
                print(chr(per))
            print("%s" % "")
            print("User Serial Number: ")
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                print(chr(per))
    print('\n')
    return deviceList



def openCamera(cameraID, deviceList, inputSize):
    global cam, stcam, imageSize
    
    if deviceList.nDeviceNum != 0:
        imageSize = inputSize
        # ch:选择设备并创建句柄 | en:Select device and create handle
        stDeviceList = cast(deviceList.pDeviceInfo[int(cameraID)], POINTER(MV_CC_DEVICE_INFO)).contents

        ret = cam.MV_CC_CreateHandle(stDeviceList)
        if ret != 0:
            print("Create Handle fail! ret[0x%x]" % ret)
            #sys.exit()

        # ch:打开设备 | en:Open device
        ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            print("Open Device fail! ret[0x%x]" % ret)
            #sys.exit()
            
        # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
        if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
            nPacketSize = cam.MV_CC_GetOptimalPacketSize()
            if int(nPacketSize) > 0:
                ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize",nPacketSize)
                if ret != 0:
                    print ("warning: set packet size fail! ret[0x%x]" % ret)
            else:
                print ("warning: set packet size fail! ret[0x%x]" % nPacketSize)
                
        # ch:设置触发模式为off | en:Set trigger mode as off
        ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            print("Set Trigger Mode fail! ret[0x%x]" % ret)
            #sys.exit()

        # ch:开始取流 | en:Start grab image
        ret = cam.MV_CC_StartGrabbing()
        if ret != 0:
            print("Start Grabbing fail! ret[0x%x]" % ret)
            #sys.exit()
            
        #time.sleep(5)
        
        # 单帧
        ret = cam.MV_CC_SetEnumValue("TriggerMode",1)
        if ret != 0:
            print(f'set triggermode fail! ret = {ret}')
        ret = cam.MV_CC_SetEnumValue("TriggerSource",7)
        if ret != 0:
            print(f'set triggersource fail! ret = {ret}')


deviceList = getCameraList()


if int(nConnectionNum) >= deviceList.nDeviceNum:
    print("Intput error!")
    #sys.exit()

openCamera(nConnectionNum, deviceList, imageSize)

stOutFrame = MV_FRAME_OUT()  
img_buff = None
buf_cache = None
numArray = None

def color_numpy(data,nWidth,nHeight):
    data_ = np.frombuffer(data, count=int(nWidth*nHeight*3), dtype=np.uint8, offset=0)
    data_r = data_[0:nWidth*nHeight*3:3]
    data_g = data_[1:nWidth*nHeight*3:3]
    data_b = data_[2:nWidth*nHeight*3:3]

    data_r_arr = data_r.reshape(nHeight, nWidth)
    data_g_arr = data_g.reshape(nHeight, nWidth)
    data_b_arr = data_b.reshape(nHeight, nWidth)
    numArray = np.zeros([nHeight, nWidth, 3],"uint8")

    numArray[:, :, 0] = data_r_arr
    numArray[:, :, 1] = data_g_arr
    numArray[:, :, 2] = data_b_arr
    return numArray

def get_image(name=None):
    global cam
    global stOutFrame
    global img_buff
    global numArray
    global buf_cache
    global imageSize
    
    try:
        ret = cam.MV_CC_ClearImageBuffer()
        if ret != 0:
            print(f"MV_CC_ClearImageBuffer fail.  ret = {ret}")
                
        ret = cam.MV_CC_SetCommandValue("TriggerSoftware")
        if ret != 0:
            print(f'set triggersoftware fail! ret = {ret}')
            for i in range(100):
                try:
                    stopCamera()
                    openCamera(nConnectionNum,getCameraList(),imageSize)
                    ret = cam.MV_CC_SetCommandValue("TriggerSoftware")
                    if ret == 0:
                        break
                except:
                    traceback.print_exc()

        ret = cam.MV_CC_GetImageBuffer(stOutFrame, 2000)
        if 0 == ret:
            if None == buf_cache:
                buf_cache = (c_ubyte * stOutFrame.stFrameInfo.nFrameLen)()
            #获取到图像的时间开始节点获取到图像的时间开始节点
            st_frame_info = stOutFrame.stFrameInfo
            cdll.msvcrt.memcpy(byref(buf_cache), stOutFrame.pBufAddr, st_frame_info.nFrameLen)
            #print ("get one frame: Width[%d], Height[%d], nFrameNum[%d]"  % (st_frame_info.nWidth, st_frame_info.nHeight, st_frame_info.nFrameNum))
            n_save_image_size = st_frame_info.nWidth * st_frame_info.nHeight * 3 + 2048
            if img_buff is None:
                img_buff = (c_ubyte * n_save_image_size)()
            #转换像素结构体赋值
            stConvertParam = MV_CC_PIXEL_CONVERT_PARAM()
            memset(byref(stConvertParam), 0, sizeof(stConvertParam))
            stConvertParam.nWidth = st_frame_info.nWidth
            stConvertParam.nHeight = st_frame_info.nHeight
            stConvertParam.pSrcData = cast(buf_cache, POINTER(c_ubyte))
            stConvertParam.nSrcDataLen = st_frame_info.nFrameLen
            stConvertParam.enSrcPixelType = st_frame_info.enPixelType 

            # RGB直接显示
            if PixelType_Gvsp_RGB8_Packed == st_frame_info.enPixelType:
                numArray = color_numpy(buf_cache,st_frame_info.nWidth,st_frame_info.nHeight)

            #如果是彩色且非RGB则转为RGB后显示
            else:
                nConvertSize = st_frame_info.nWidth * st_frame_info.nHeight * 3
                stConvertParam.enDstPixelType = PixelType_Gvsp_RGB8_Packed
                stConvertParam.pDstBuffer = (c_ubyte * nConvertSize)()
                stConvertParam.nDstBufferSize = nConvertSize
                #time_start=time.time()
                ret = cam.MV_CC_ConvertPixelType(stConvertParam)
                #time_end=time.time()
                #print('MV_CC_ConvertPixelType:',time_end - time_start) 
                if ret != 0:
                    print(f'convert pixel fail! ret = {ret}')

                cdll.msvcrt.memcpy(byref(img_buff), stConvertParam.pDstBuffer, nConvertSize)
                numArray = color_numpy(img_buff,st_frame_info.nWidth,st_frame_info.nHeight)
            
            p = pil_image.fromarray(numArray)
            #p = p.resize(imageSize,pil_image.NEAREST)
            numArray = np.asarray(p)
            #numArray = copy.deepcopy(cv2.resize(numArray, imageSize))
            numArray = cv2.cvtColor(numArray, cv2.COLOR_BGR2RGB)
            nRet = cam.MV_CC_FreeImageBuffer(stOutFrame)
            
        else:
            print(f"no data, nret = {ret}")
            try:
                stopCamera()
                #mb.showwarning('Hikrobot Camera Error', 'Camera disconnected 3 times')
                openCamera(nConnectionNum,getCameraList(),imageSize)
            except:
                traceback.print_exc()
            
        frame = {"2":numArray}
        if name is not None:
            return frame[name]
        return frame
    except:
        traceback.print_exc()
        print("hik retrive image failed")
        return {"2":None}

