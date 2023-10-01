import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import math
from FMCV import Logging
import traceback
# SIFT
def angle_cosine(p0, p1, p2):
    d1, d2 = p0 - p1, p2 - p1
    return abs(np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2)))

def is_rectangle(points, angle_threshold=0.20):
    p0, p1, p2, p3 = points.squeeze()
    cosines = [angle_cosine(p0, p1, p2),
               angle_cosine(p1, p2, p3),
               angle_cosine(p2, p3, p0),
               angle_cosine(p3, p0, p1)]

    for cosine in cosines:
        if abs(cosine) > angle_threshold:
            return False

    return True

# Initiate SIFT detector
sift = cv2.SIFT_create()
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

def match_rotate(template, frame, angle_threshold = 0.2, disable_rectangle_check = False, blur = 0):
    try:
    
        if blur >0:
            template  = cv2.blur(template,(blur, blur))
            frame = cv2.blur(frame,(blur, blur))
    
        # Find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(cv2.cvtColor(template, cv2.COLOR_BGR2GRAY), None)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        kp2, des2 = sift.detectAndCompute(gray, None)

        matches = flann.knnMatch(des1, des2, k=2)

        # Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        if len(good_matches) > 10:            
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC , 3)
            inv_M = np.linalg.inv(M)
            matches_mask = mask.ravel().tolist()

            h, w = template.shape[:2]
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            if M is not None:
                dst = cv2.perspectiveTransform(pts, M)        
                if is_rectangle(dst, angle_threshold) or disable_rectangle_check:
                    cropped = cv2.warpPerspective(frame, inv_M, (w, h))
                    
                    frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

                    xy_dst = cv2.perspectiveTransform(np.float32([[int(w/2),int(h/2)],[int(w/2),0]]).reshape(-1, 1, 2), M)
                    x_center = int(xy_dst[0][0][0])
                    y_center = int(xy_dst[0][0][1])
                                
                    cv2.circle(frame, (x_center, y_center), 3, (255, 0, 0), -1)
                    angle = -np.degrees(np.arctan2(xy_dst[0][0][0] - xy_dst[1][0][0] , xy_dst[0][0][1] - xy_dst[1][0][1]))
                    
                    cv2.putText(frame, "Angle: {:.2f} degrees".format(angle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                    cv2.arrowedLine(frame, (x_center, y_center),(int(xy_dst[1][0][0]),int(xy_dst[1][0][1])), (255, 0, 0), 2, cv2.LINE_AA)
                    return frame, cropped, [x_center, y_center], angle , dst, xy_dst
    except:
        Logging.debug(traceback.format_exc())

def extract_rotated_rectangle(image, center, width, height, angle):

    # Rotate the image by the inverse of the given angle
    M = cv2.getRotationMatrix2D(center, angle, 1)  # Now using the positive angle
    rotated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # Extract the rectangle from the rotated image
    x_c, y_c = center
    x_start, y_start = int(x_c - width/2), int(y_c - height/2)
    x_end, y_end = int(x_c + width/2), int(y_c + height/2)
    cropped = rotated_image[y_start:y_end, x_start:x_end]

    return cropped
    
    
def rotated_rectangle_points(w, h, x_c, y_c, theta):
    # Convert degrees to radians
    theta_rad = math.radians(theta)
    
    # Define the original points
    P1 = (-w/2, -h/2)
    P2 = (w/2, -h/2)
    P3 = (w/2, h/2)
    P4 = (-w/2, h/2)

    # Rotate the points
    def rotate_point(x, y, theta_rad):
        x_prime = x * math.cos(theta_rad) - y * math.sin(theta_rad)
        y_prime = x * math.sin(theta_rad) + y * math.cos(theta_rad)
        return x_prime, y_prime

    P1_rot = rotate_point(P1[0], P1[1], theta_rad)
    P2_rot = rotate_point(P2[0], P2[1], theta_rad)
    P3_rot = rotate_point(P3[0], P3[1], theta_rad)
    P4_rot = rotate_point(P4[0], P4[1], theta_rad)

    # Translate the points to the given center
    P1_final = (P1_rot[0] + x_c, P1_rot[1] + y_c)
    P2_final = (P2_rot[0] + x_c, P2_rot[1] + y_c)
    P3_final = (P3_rot[0] + x_c, P3_rot[1] + y_c)
    P4_final = (P4_rot[0] + x_c, P4_rot[1] + y_c)

    return P1_final, P2_final, P3_final, P4_final
    
def distance(p1, p2):
    """Calculate distance between two points."""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def calculate_distance(num1, num2):
    return abs(num1 - num2)

def compute_dimensions(p1, p2, p3, p4):
    """Find width and height of a rectangle given its 4 points."""
    # Calculate pairwise distances
    distances = [
        distance(p1, p2),
        distance(p1, p3),
        distance(p1, p4),
        distance(p2, p3),
        distance(p2, p4),
        distance(p3, p4)
    ]

    # Sort the distances
    distances.sort()

    # The largest two distances are the diagonals. We don't need them.
    # The next two largest are the width and height, but we can't be sure which is which.
    # Therefore, we return them both.
    width = distances[-3]
    height = distances[-4]

    return min(width, height), max(width, height)    

def rectangle_dimensions(x1, y1, x2, y2):
    width = x2 - x1
    height = y2 - y1
    return abs(width), abs(height)

box_points = []
button_down = False

def rotate_image_(image, angle):
    h, w = image.shape[:2]
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, -angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    pixel_array = np.full((h, w, 1), (255), dtype=np.uint8)
    mask = cv2.warpAffine(pixel_array, rot_mat, image.shape[1::-1])

    #cv2.imshow("1",result)
    #cv2.waitKey(0)
    return result,mask,w,h

def rotate_image(image, angle):
    h, w = image.shape[:2]
    cx, cy = (w // 2, h // 2)

    # get rotation matrix (explained in section below)
    M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)

    # get cos and sin value from the rotation matrix
    cos, sin = abs(M[0, 0]), abs(M[0, 1])

    # calculate new width and height after rotation (explained in section below)
    newW = int((h * sin) + (w * cos))
    newH = int((h * cos) + (w * sin))

    # calculate new rotation center
    M[0, 2] += (newW / 2) - cx
    M[1, 2] += (newH / 2) - cy

    # use modified rotation center and rotation matrix in the warpAffine method
    result = cv2.warpAffine(image, M, (newW, newH), borderValue=(0,0,0),flags=cv2.INTER_LINEAR) 

    pixel_array = np.full((h, w, 1), (255), dtype=np.uint8)
    mask = cv2.warpAffine(pixel_array, M, (newW, newH)) 

    #print(M)
    # cv2.imshow("2",result)
    #cv2.imshow("g",mask)
    # cv2.waitKey(0)
    
    return result,mask,newW,newH

def scale_image(image, percent, maxwh):
    max_width = maxwh[1]
    max_height = maxwh[0]
    max_percent_width = max_width / image.shape[1] * 100
    max_percent_height = max_height / image.shape[0] * 100
    max_percent = 0
    if max_percent_width < max_percent_height:
        max_percent = max_percent_width
    else:
        max_percent = max_percent_height
    if percent > max_percent:
        percent = max_percent
    width = int(image.shape[1] * percent / 100)
    height = int(image.shape[0] * percent / 100)
    result = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)
    return result, percent

def click_and_crop(event, x, y, flags, param):
    global box_points, button_down
    if (button_down == False) and (event == cv2.EVENT_LBUTTONDOWN):
        button_down = True
        box_points = [(x, y)]
    elif (button_down == True) and (event == cv2.EVENT_MOUSEMOVE):
        image_copy = param.copy()
        point = (x, y)
        cv2.rectangle(image_copy, box_points[0], point, (0, 255, 0), 2)
        cv2.imshow("Template Cropper - Press C to Crop", image_copy)
    elif event == cv2.EVENT_LBUTTONUP:
        button_down = False
        box_points.append((x, y))
        cv2.rectangle(param, box_points[0], box_points[1], (0, 255, 0), 2)
        cv2.imshow("Template Cropper - Press C to Crop", param)

# GUI template cropping tool
def template_crop(image):
    clone = image.copy()
    cv2.namedWindow("Template Cropper - Press C to Crop", cv2.WINDOW_NORMAL)
    param = image.copy()
    cv2.setMouseCallback("Template Cropper - Press C to Crop", click_and_crop, param)
    while True:
        cv2.imshow("Template Cropper - Press C to Crop", param)
        key = cv2.waitKey(1)
        if key == ord("c"):
            cv2.destroyAllWindows()
            break
    if len(box_points) == 2:
        cropped_region = clone[box_points[0][1]:box_points[1][1], box_points[0][0]:box_points[1][0]]
    return cropped_region

def modifiedMatchTemplate(rgbimage, rgbtemplate, method, matched_thresh, rgbdiff_thresh, rot_range, rot_interval, scale_range, scale_interval, rm_redundant, minmax):
    """
    rgbimage: RGB image where the search is running.
    rgbtemplate: RGB searched template. It must be not greater than the source image and have the same data type.
    method: [String] Parameter specifying the comparison method
    matched_thresh: [Float] Setting threshold of matched results(0~1).
    rgbdiff_thresh: [Float] Setting threshold of average RGB difference between template and source image.
    rot_range: [Integer] Array of range of rotation angle in degrees. Example: [0,360]
    rot_interval: [Integer] Interval of traversing the range of rotation angle in degrees.
    scale_range: [Integer] Array of range of scaling in percentage. Example: [50,200]
    scale_interval: [Integer] Interval of traversing the range of scaling in percentage.
    rm_redundant: [Boolean] Option for removing redundant matched results based on the width and height of the template.
    minmax:[Boolean] Option for finding points with minimum/maximum value.

    Returns: List of satisfied matched points in format [[point.x, point.y], angle, scale].
    """
    image_maxwh = rgbimage.shape
    print(image_maxwh)
    height, width = rgbtemplate.shape
    all_points = []
    if minmax == False:
        for next_angle in range(rot_range[0], rot_range[1], rot_interval):
            for next_scale in range(scale_range[0], scale_range[1], scale_interval):
                scaled_template, actual_scale = scale_image(rgbtemplate, next_scale, image_maxwh)
                if next_angle == 0:
                    h, w = scaled_template.shape[:2]
                    mask = np.full((h, w, 1), (255), dtype=np.uint8)
                    rotated_template = scaled_template
                    x, y = w, h
                else:
                    rotated_template, mask, x, y = rotate_image(scaled_template, next_angle)
                    
                if method == "TM_CCOEFF":
                    matched_points = cv2.matchTemplate(rgbimage,rotated_template,cv2.TM_CCOEFF, None, mask)
                    satisfied_points = np.where(matched_points >= matched_thresh)
                elif method == "TM_CCOEFF_NORMED":
                    matched_points = cv2.matchTemplate(rgbimage,rotated_template,cv2.TM_CCOEFF_NORMED, None, mask)
                    satisfied_points = np.where(matched_points >= matched_thresh)
                elif method == "TM_CCORR":
                    matched_points = cv2.matchTemplate(rgbimage,rotated_template,cv2.TM_CCORR, None, mask)
                    satisfied_points = np.where(matched_points >= matched_thresh)
                elif method == "TM_CCORR_NORMED":
                    matched_points = cv2.matchTemplate(rgbimage,rotated_template,cv2.TM_CCORR_NORMED, None, mask)
                    satisfied_points = np.where(matched_points >= matched_thresh)
                elif method == "TM_SQDIFF":
                    matched_points = cv2.matchTemplate(rgbimage,rotated_template,cv2.TM_SQDIFF, None, mask)
                    satisfied_points = np.where(matched_points <= matched_thresh)
                elif method == "TM_SQDIFF_NORMED":
                    matched_points = cv2.matchTemplate(rgbimage,rotated_template,cv2.TM_SQDIFF_NORMED, None, mask)
                    satisfied_points = np.where(matched_points <= matched_thresh)
                else:
                    raise Exception("There's no such comparison method for template matching.")
                for pt in zip(*satisfied_points[::-1]):
                    all_points.append([pt, next_angle, actual_scale])
    else:
        stop = False
        for next_angle in range(rot_range[0], rot_range[1], rot_interval):
            if stop:
                break
            for next_scale in range(scale_range[0], scale_range[1], scale_interval):
                scaled_template, actual_scale = scale_image(rgbtemplate, next_scale, image_maxwh)
                
                if next_angle == 0:
                    h,w = scaled_template.shape[:2]
                    mask = np.full((h, w, 1), (255), dtype=np.uint8)
                    rotated_template = scaled_template
                    x,y =w,h
                else:
                    rotated_template, mask, x, y  = rotate_image(scaled_template, next_angle)
                if method == "TM_CCOEFF":
                    matched_points = cv2.matchTemplate(rgbimage,rotated_template,cv2.TM_CCOEFF, None, mask)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
                    max_loc = (max_loc[0],max_loc[1])
                    if max_val >= matched_thresh:
                        all_points.append([max_loc, next_angle, actual_scale, max_val,x,y])
                        
                elif method == "TM_CCOEFF_NORMED":    
                    #cv2.imshow("m",mask)
                    #cv2.imshow("2",rgbimage)
                    #cv2.imshow("1",rotated_template)
                    #cv2.waitKey(0)                
                    matched_points = cv2.matchTemplate(rgbimage,rotated_template,cv2.TM_CCOEFF_NORMED, None, mask)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
                    if max_val >= matched_thresh and not math.isinf(max_val):
                        all_points.append([max_loc, next_angle, actual_scale, max_val,x,y])
                    #print([max_loc, next_angle, actual_scale, max_val, x, y])
                elif method == "TM_CCORR":
                    matched_points = cv2.matchTemplate(rgbimage,rotated_template,cv2.TM_CCORR, None, mask)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
                    if max_val >= matched_thresh:
                        all_points.append([max_loc, next_angle, actual_scale, max_val,x,y])
                elif method == "TM_CCORR_NORMED":
                    matched_points = cv2.matchTemplate(rgbimage,rotated_template,cv2.TM_CCORR_NORMED, None, mask)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
                    if max_val >= matched_thresh:
                        all_points.append([max_loc, next_angle, actual_scale, max_val,x,y])
                        #stop = True
                        #break
                elif method == "TM_SQDIFF":
                    matched_points = cv2.matchTemplate(rgbimage,rotated_template,cv2.TM_SQDIFF, None, mask)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
                    if min_val <= matched_thresh:
                        all_points.append([min_loc, next_angle, actual_scale, min_val,x,y])
                elif method == "TM_SQDIFF_NORMED":
                    matched_points = cv2.matchTemplate(rgbimage,rotated_template,cv2.TM_SQDIFF_NORMED, None, mask)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
                    if min_val <= matched_thresh:
                        all_points.append([min_loc, next_angle, actual_scale, min_val,x,y])
                else:
                    raise Exception("There's no such comparison method for template matching.")
                    
                    
        if method == "TM_CCOEFF":
            all_points = sorted(all_points, key=lambda x: -x[3])
        elif method == "TM_CCOEFF_NORMED":
            all_points = sorted(all_points, key=lambda x: -x[3])
        elif method == "TM_CCORR":
            all_points = sorted(all_points, key=lambda x: -x[3])
        elif method == "TM_CCORR_NORMED":
            all_points = sorted(all_points, key=lambda x: -x[3])
        elif method == "TM_SQDIFF":
            all_points = sorted(all_points, key=lambda x: x[3])
        elif method == "TM_SQDIFF_NORMED":
            all_points = sorted(all_points, key=lambda x: x[3])
    if rm_redundant == True:
        lone_points_list = []
        visited_points_list = []
        for point_info in all_points:
            point = point_info[0]
            scale = point_info[2]
            width = point_info[4]
            height = point_info[5]
            all_visited_points_not_close = True
            if len(visited_points_list) != 0:
                for visited_point in visited_points_list:
                    if ((abs(visited_point[0] - point[0]) < (width * scale / 100)) and (abs(visited_point[1] - point[1]) < (height * scale / 100))):
                        all_visited_points_not_close = False
                if all_visited_points_not_close == True:
                    lone_points_list.append(point_info)
                    visited_points_list.append(point)
            else:
                lone_points_list.append(point_info)
                visited_points_list.append(point)
        points_list = lone_points_list
    else:
        points_list = all_points
    return points_list


def main():
    img_bgr = cv2.imread('target.png')
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    template_bgr = cv2.imread('templ.png')
    cropped_template_bgr = template_bgr
    #cropped_template_bgr = template_crop(template_bgr)
    cropped_template_rgb = cv2.cvtColor(cropped_template_bgr, cv2.COLOR_BGR2RGB)
    cropped_template_rgb = np.array(cropped_template_rgb)
    cropped_template_gray = cv2.cvtColor(cropped_template_rgb, cv2.COLOR_RGB2GRAY)
    h, w = cropped_template_gray.shape
    fig = plt.figure(num='Template - Close the Window to Continue >>>')
    plt.imshow(cropped_template_rgb)
    #plt.show()
    b= 10
    points_list = modifiedMatchTemplate(cv2.blur(img_gray,(10,10)), cv2.blur(cropped_template_gray,(10,10)), "TM_CCOEFF_NORMED",  0.6, 255, [0,360], 3, [100,101], 2, True, True)
    fig, ax = plt.subplots(1)
    plt.gcf().canvas.set_window_title('Template Matching Results')
    ax.imshow(img_rgb)
    centers_list = []
    for point_info in points_list:
        point = point_info[0]
        angle = point_info[1]
        scale = point_info[2]
        width = point_info[4]
        height = point_info[5]
        centers_list.append([point, scale])
        center_x = point[0] + (width/2)*scale/100
        center_y = point[1] + (height/2)*scale/100
        plt.scatter(center_x, center_y, s=20, color="red")
        plt.scatter(point[0], point[1], s=20, color="green")
        rectangle = patches.Rectangle((center_x - w/2, center_y - h/2), w*scale/100, h*scale/100, color="red", alpha=0.50, label='Matched box')
        box = patches.Rectangle((point[0], point[1]), width*scale/100, height*scale/100, color="green", alpha=0.50, label='Bounding box')
        transform = mpl.transforms.Affine2D().rotate_deg_around(point[0] + width/2*scale/100, point[1] + height/2*scale/100, angle) + ax.transData
        rectangle.set_transform(transform)
        ax.add_patch(rectangle)
        ax.add_patch(box)
        plt.legend(handles=[rectangle,box])
        
        print(f"{angle}")
    #plt.grid(True)
    plt.show()
    fig2, ax2 = plt.subplots(1)
    plt.gcf().canvas.set_window_title('Template Matching Results')
    ax2.imshow(img_rgb)
    for point_info in centers_list:
        point = point_info[0]
        scale = point_info[1]
        plt.scatter(point[0]+width/2*scale/100, point[1]+height/2*scale/100, s=20, color="red")
    plt.show()


if __name__ == "__main__":
    main()
