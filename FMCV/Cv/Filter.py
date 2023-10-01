import cv2
import numpy as np


def pcb_1(image):
    '''
        return a binary image for pcb 2d matrix on dot printed dyson product
    '''

    # Apply adaptive thresholding
    # cv2.ADAPTIVE_THRESH_MEAN_C means threshold value is the mean of neighbourhood area
    # cv2.ADAPTIVE_THRESH_GAUSSIAN_C means threshold value is the weighted sum of neighbourhood values where weights are a Gaussian window
    # Block size decides the size of neighbourhood area and C is a constant subtracted from the mean or weighted mean
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray_image, (7, 7), 0)

    thresh = cv2.adaptiveThreshold(blurred, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,201, 10)

    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)
    # Filter out small dots based on area in stats
    min_size = 1800  # Minimum component size
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_size:
            thresh = np.where(labels_im == i,0, thresh)

    kernel = np.ones((3,3),np.uint8)
    #dilation = cv2.dilate(result,kernel,iterations = 1)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return closing
    
def ostu_binary(image):
    if len(image.shape) ==3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    ret,image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    return image

def CLAHE(image):
    '''Contrast Limited Adaptive Histogram Equalization'''
    if len(image.shape) ==3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(image)
    
    return cl1

    


    