import cv2

print("Loading Camera Python Script")

cm1 = cv2.VideoCapture(1,cv2.CAP_DSHOW)
#cm1 = cv2.VideoCapture(0,cv2.CAP_DSHOW)
#cm1.set(cv2.CAP_PROP_FRAME_WIDTH,2592)
#cm1.set(cv2.CAP_PROP_FRAME_HEIGHT,1944)
# cm2 = cv2.VideoCapture(1)
# cm2.set(cv2.CAP_PROP_FRAME_WIDTH,2595)
# cm2.set(cv2.CAP_PROP_FRAME_HEIGHT,1944)
#cm2 = cv2.VideoCapture(1,cv2.CAP_DSHOW)
# cm1.set(cv2.CAP_PROP_FRAME_WIDTH,3264)
# cm1.set(cv2.CAP_PROP_FRAME_HEIGHT,2448)
#cm1.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
#cm1.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
cm1.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cm1.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
cm1.set(cv2.CAP_PROP_AUTOFOCUS,1)
#cm1.set(cv2.CAP_PROP_FOCUS,450)

ret, frm1 = cm1.read()
#ret, frm2 = cm2.read()
def get_image(name=None):
    ret, frm1 = cm1.read()
    #ret, frm2 = cm2.read()
    #frm2 = cv2.flip(frm2, 0)
    #return frm1
    frames = {"1":frm1}
    #frames = {"0":frm1,"1":frm2}
    #frames = {"0":frm1,"2":cv2.flip(frm1, 0),"3":cv2.flip(frm1, 1)}
    if name is not None:
       return frames[name]
    return frames