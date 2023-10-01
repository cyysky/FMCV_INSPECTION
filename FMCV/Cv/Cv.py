import cv2
import numpy as np
from FMCV import Logging

from PIL import Image, ImageOps



def resize_maintain_ratio_by_width(image,new_width):

    # Calculate the ratio of the new width to the old width
    ratio = new_width / image.shape[1]
    new_height = int(image.shape[0] * ratio)
    
    # Perform the actual resizing of the image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation = cv2.INTER_LANCZOS4)
    
    return resized_image


def image_make_square_border_pil(img):
    # Open the image using PIL
    #img = Image.open(image)

    # Get image dimensions
    width, height = img.size

    # Find the maximum dimension
    max_dim = max(width, height)

    # Calculate border sizes
    top = int((max_dim - height) / 2)
    bottom = max_dim - height - top
    left = int((max_dim - width) / 2)
    right = max_dim - width - left

    # Create a new square image with a black border
    square_image = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))

    # Paste the original image onto the square image with the calculated borders
    square_image.paste(img, (left, top))

    return square_image

def image_make_square_border(image):
    # Get image dimensions
    height, width = image.shape[:2]

    # Find the maximum dimension
    max_dim = max(height, width)

    # Calculate border sizes
    top = int((max_dim - height) / 2)
    bottom = max_dim - height - top
    left = int((max_dim - width) / 2)
    right = max_dim - width - left

    # Create the square image with centered content
    square_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    return square_image

def get_rotate(name, src):
    if name.strip() == "":
        return src
        
    if name.startswith("F"):
        src = cv2.flip(src, 1)
        
    if name in ("90","F90"):
        return cv2.rotate(src, cv2.ROTATE_90_CLOCKWISE)
        
    elif name in ("180","F180"):
        return cv2.rotate(src, cv2.ROTATE_180)
        
    elif name in ("270","F270"):
        return cv2.rotate(src, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
    return src
    
def get_rotate_pil(name, src):
    # Convert PIL Image to NumPy array (image in RGB format)
    cv2_image = np.array(src)

    # Convert RGB to BGR 
    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)
    
    # Rotate the image
    cv2_image = get_rotate(name, cv2_image)
    
    # Convert RGB to BGR 
    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

    return Image.fromarray(cv2_image)

def get_rotate_rectangle(name, imshape, bbox):
    #bbox = (x1,y1,x2,y2)
    #imshape = (w,h)
    if name.strip() == "":
        return bbox
    
    w, h = imshape[0],imshape[1]
    x1, y1, x2, y2 = bbox[0],bbox[1],bbox[2],bbox[3]
    
    if name == "F":
        x1 = (w-1)-x1
        x2 = (w-1)-x2
        return (x1,y1,x2,y2)
        
    if name == "F180":
        x1 = (w-1)-x1
        x2 = (w-1)-x2

    if name in ("90","F90"):
        angle = 90
        w,h = h, w
        
    elif name in ("180","F180"):
        angle = 180
        
    elif name in ("270","F270"):
        angle = -90
        w,h = h, w
        
    cx, cy = (int(w / 2), int(h / 2))

    bbox_tuple = [
            (x1, y1),
            (x2, y1),
            (x2, y2),
            (x1, y2),
        ] # put x and y coordinates in tuples, we will iterate through the tuples and perform rotation

    rotated_bbox = []

    for i, coord in enumerate(bbox_tuple):

      M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
      cos, sin = abs(M[0, 0]), abs(M[0, 1])
      newW = int((h * sin) + (w * cos))
      newH = int((h * cos) + (w * sin))
      M[0, 2] += (newW / 2) - cx
      M[1, 2] += (newH / 2) - cy
      v = [coord[0], coord[1], 1]
      adjusted_coord = np.dot(M, v)
      rotated_bbox.insert(i, (adjusted_coord[0], adjusted_coord[1]))

    result = [int(x) for t in rotated_bbox for x in t]

    xmin, ymin, xmax, ymax = min(result[0],result[4]), min(result[1],result[5]), max(result[0],result[4]), max(result[1],result[5])

    if name in ("F90","F270"):
        xmin = (h-1)-xmin
        xmax = (h-1)-xmax
        
    Logging.debug(f"xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")
    
    return (xmin, ymin, xmax, ymax)


def _show_im(frm):
    cv2.namedWindow("a", cv2.WINDOW_NORMAL)
    cv2.imshow("a",frm)        
    cv2.waitKey(0)

def to_gray(CurrentImage):
    # Convert to grayscale. 
    if len(CurrentImage.shape)==3:
        if CurrentImage.shape[2] == 3:
            return cv2.cvtColor(CurrentImage, cv2.COLOR_BGR2GRAY) 
    return CurrentImage 
    
def to_color(CurrentImage):
    # Convert to grayscale. 
    if len(CurrentImage.shape)<3:
        img1 = cv2.cvtColor(CurrentImage, cv2.COLOR_GRAY2BGR)
    else:
        img1 = CurrentImage
    return img1

def match_template(TemplateImage,CurrentImage,Mode = 5, Mask = None, BlurSize = 30):
    #Reference https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html
    #Get width and height
    w = TemplateImage.shape[1]
    h = TemplateImage.shape[0]
    
    #if Config.TEMPLATE_BLUR>0:
    #    TemplateImage = cv2.blur(TemplateImage,(30, 30))
    #    CurrentImage = cv2.blur(CurrentImage,(30, 30))
    
    #https://docs.opencv.org/4.x/df/dfb/group__imgproc__object.html#ga3a7850640f1fe1f58fe91a2d7583695d
    #Mode = 5 - 1
    # All the 6 methods for comparison in a list
    methods = [ 'cv2.TM_SQDIFF', 
                'cv2.TM_SQDIFF_NORMED',
                'cv2.TM_CCORR',
                'cv2.TM_CCORR_NORMED', 
                'cv2.TM_CCOEFF', 
                'cv2.TM_CCOEFF_NORMED']
    method = eval(methods[Mode])
    
    # Template matching
    #res = cv2.matchTemplate(to_gray(CurrentImage),to_gray(TemplateImage),method)
    
    if BlurSize > 0:
        frame  = cv2.blur(to_gray(CurrentImage),(BlurSize, BlurSize))
        template = cv2.blur(to_gray(TemplateImage),(BlurSize, BlurSize))
    else:
        frame  = to_gray(CurrentImage)
        template = to_gray(TemplateImage)
        
    if Mask is not None:
        print("detect with mask")
        res = cv2.matchTemplate(frame, template, method, None, Mask)
    else:
        res = cv2.matchTemplate(frame, template, method)
    #cv2.imshow("1",cv2.blur(to_gray(CurrentImage),(BlurSize, BlurSize)))
    #cv2.waitKey(100)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)    
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
        top_val = min_val
    else:
        top_left = max_loc
        top_val = max_val
    bottom_right = (top_left[0] + w, top_left[1] + h)
    loc = np.where(res>=0.6)

    croppedImage = CurrentImage[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]].copy()

    #cv2.putText(croppedImage,str(top_val),(-10,11),cv2.FONT_HERSHEY_SIMPLEX,0.33,(0,0,255),1)

    return res,top_left,bottom_right,top_val,croppedImage  
    
    
def get_white_image_with_text(text, image_height = 480 , image_width = 640):
    #image_height = 480
    #image_width = 640
    number_of_color_channels = 3
    color = (255,255,255)
    pixel_array = np.full((image_height, image_width, number_of_color_channels), color, dtype=np.uint8)
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX 
    # org
    org = (50, 50)
    # fontScale
    fontScale = 1
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 2
    # Using cv2.putText() method
    #time.sleep(100)
    pixel_array = cv2.putText(pixel_array, text, org, font, 
                       fontScale, color, thickness, cv2.LINE_AA)

    return pixel_array
    
def non_max_suppression(boxes, overlapThresh):
    if len(boxes) == 0:
        return []

    pick = []
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick]

    
def get_matching_positions(img, template, meth = 5 , threshold = 0.7 , mask = None, blur_size = 10, nms_threshold = 0.5):
    h, w = template.shape[:2]
    
    #https://docs.opencv.org/4.x/df/dfb/group__imgproc__object.html#ga3a7850640f1fe1f58fe91a2d7583695d
    #Mode = 5 - 1
    # All the 6 methods for comparison in a list
    methods = [ 'cv2.TM_SQDIFF', 
                'cv2.TM_SQDIFF_NORMED',
                'cv2.TM_CCORR',
                'cv2.TM_CCORR_NORMED', 
                'cv2.TM_CCOEFF', 
                'cv2.TM_CCOEFF_NORMED']
    method = eval(methods[meth])

    all_positions = {}
    
    # Apply template Matching
    if blur_size > 0:
        frame  = cv2.blur(to_gray(img),(blur_size, blur_size))
        template = cv2.blur(to_gray(template),(blur_size, blur_size))
    else:
        frame  = to_gray(img)
        template = to_gray(template)
        
    if mask is not None:
        print("detect with mask")
        res = cv2.matchTemplate(frame, template, method, None, mask)
    else:
        res = cv2.matchTemplate(frame, template, method)

    # Depending on the method, the way to extract matches above threshold differs
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        all_positions = np.where(res <= threshold)
    else:
        all_positions = np.where(res >= threshold)

    # Apply NMS
    points = list(zip(*all_positions[::-1]))
    boxes = np.array([[pt[0], pt[1], pt[0] + w, pt[1] + h] for pt in points])
    pick = non_max_suppression(boxes, nms_threshold)
    
    return pick