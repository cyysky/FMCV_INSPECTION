import cv2
import copy
import traceback

from FMCV.Cv import Cv,Filter

qr_detector = cv2.QRCodeDetector()

try:
    from pyzbar.pyzbar import decode
    from pylibdmtx.pylibdmtx import decode as dm_decode
except:
    traceback.print_exc()
    
def process_barcode(self, result, frm, src_n, step_n, roi_n):

    h,w = frm.shape[:2]
    roi = self.Main.results[src_n][step_n][roi_n]
    result = roi

    m = roi['margin']
    x1 = roi['x1'] - m
    y1 = roi['y1'] - m
    x2 = roi['x2'] + m
    y2 = roi['y2'] + m
    cropped = frm[y1:y2,x1:x2]
    
    result.update({"CODE":""})
    result.update({'result_image':copy.deepcopy(cropped)})
    
    if roi['qr_filter'] == "PCB_1":
        cropped = Filter.pcb_1(cropped)
    elif roi['qr_filter'] == "EQUALIZE":
        cropped = Filter.CLAHE(cropped)
    elif roi['qr_filter'] == "BINARY":
        cropped = Filter.ostu_binary(cropped)
            
    if roi['qr_size'] > 10:
        cropped = Cv.resize_maintain_ratio_by_width(cropped, roi['qr_size'])

    is_pass = False
    
    if 'decode' in globals():
        detectedBarcodes = decode(cropped)
        if not detectedBarcodes:
            #result.update({"PASS":False})
            print("Barcode/QR Code Not Detected or your barcode is blank/corrupted!")
        else:
            # Traverse through all the detected barcodes in image
            for barcode in detectedBarcodes:
                if barcode.data != "":               
                # Print the barcode data
                    is_pass = True
                    result.update({"CODE":barcode.data.decode("utf-8")})
                    print(barcode.data)
                    print(barcode.type)                        
    else:
        res,points, rectifiedImg = qr_detector.detectAndDecode(cropped)   
        # Detected outputs.
        if len(res) > 0:            
            print('Output : ', res[0])
            print('Bounding Box : ', points)
            result.update({"PASS":True})
            result.update({"CODE":res[0]})
        else:
            result.update({"PASS":False})
            print('QRCode not detected')

    # 2d Matrix reading
    if roi['qr_matrix']:
        print("Checking 2d matrix")
        if 'dm_decode' in globals():
        
            #cv2.imshow('image', cropped)
            #cv2.waitKey(0)
            
            cropped_dmtx = cropped
            if len(cropped_dmtx.shape) ==3:
                cropped_dmtx = cv2.cvtColor(cropped_dmtx, cv2.COLOR_BGR2GRAY)

            detectedBarcodes = dm_decode(cropped_dmtx,max_count=1)#,threshold=50)

            if not detectedBarcodes:
                print(detectedBarcodes)    
                print("Datamatrix Not Detected or your barcode is blank/corrupted!")
            else:
                # Traverse through all the detected barcodes in image
                for barcode in detectedBarcodes:
                    if barcode.data != "":               
                    # Print the barcode data
                        is_pass = True
                        result.update({"CODE":barcode.data.decode("utf-8")})
                        print(barcode.data)
                        #print(barcode.type) 
                        
    result.update({"PASS":is_pass})
    result.update({'result_image':copy.deepcopy(cropped)})
    
    return result