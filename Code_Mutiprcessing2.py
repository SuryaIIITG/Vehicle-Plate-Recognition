import cv2
import multiprocessing as mp
#import torch
import os
from PIL import Image
import time
import numpy as np
import pytesseract as pt
#import easyocr
#reader = easyocr.Reader(['en'],gpu=True)

pt.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# settings
INPUT_WIDTH =  640
INPUT_HEIGHT = 640 

#os.chdir(r"C:\Users\dhruv\My Drive\Projects\Insyst Labs\IOCL - Number Plate\Data Received")

#os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"     # Replace udp with tcp
#vcap = cv2.VideoCapture("rtsp://admin:L21F7FA6@192.168.2.76:554/cam/realmonitor?channel=1&subtype=0", cv2.CAP_FFMPEG)

# LOAD YOLO MODEL
net = cv2.dnn.readNetFromONNX('best.onnx') 
#net = cv2.dnn.readNetFromONNXTensorflow('best.pb') 
#net = cv2.dnn.readNet('best.pt')
#net = cv2.dnn.readNetFromTensorflow('best-fp16.tflite')     
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
print("Model Executed" ,net) 


# All function  
def get_detections(img,net):
    # CONVERT IMAGE TO YOLO FORMAT
    image = img.copy()
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    row, col, d = image.shape

    max_rc = max(row,col) 
    input_image = np.zeros((max_rc,max_rc,3),dtype=np.uint8)
    input_image[0:row,0:col] = image  

    # GET PREDICTION FROM YOLO MODEL 
    blob = cv2.dnn.blobFromImage(input_image,1/255,(INPUT_WIDTH,INPUT_HEIGHT),swapRB=True,crop=False)
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]
    
    return input_image, detections

def non_maximum_supression(input_image,detections):
    # FILTER DETECTIONS BASED ON CONFIDENCE AND PROBABILIY SCORE
    # center x, center y, w , h, conf, proba
    boxes = []
    confidences = []

    image_w, image_h = input_image.shape[:2]
    x_factor = image_w/INPUT_WIDTH
    y_factor = image_h/INPUT_HEIGHT

    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4] # confidence of detecting license plate
        if confidence > 0.6:
            class_score = row[5] # probability score of license plate
            if class_score > 0.25:
                cx, cy , w, h = row[0:4]

                left = int((cx - 0.5*w)*x_factor)
                top = int((cy-0.5*h)*y_factor)
                width = int(w*x_factor)
                height = int(h*y_factor)
                box = np.array([left,top,width,height]) 

                confidences.append(confidence)
                boxes.append(box)

    # clean
    boxes_np = np.array(boxes).tolist() 
    #print("Boxes : ",boxes_np)
    confidences_np = np.array(confidences).tolist()
    # NMS
    index = cv2.dnn.NMSBoxes(boxes_np,confidences_np,0.55,0.75)  #.flatten()
    
    return boxes_np, confidences_np, index

def extract_text(image,bbox):
    x,y,w,h = bbox 
    roi = image[y:y+h, x:x+w]
    
    if 0 in roi.shape: 
        return ''
    
    else:
        text = pt.image_to_string(roi, lang='eng', config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')# --oem 3
        # This is for easyocr technique 
        #text=reader.readtext(roi) 
        text = text.strip()
        print("Text of plate is: ",text)
        return text

def drawings(image,boxes_np,confidences_np,index):
    # drawings
    for ind in index:
        x,y,w,h =  boxes_np[ind]
        bb_conf = confidences_np[ind]
        conf_text = 'plate: {:.0f}%'.format(bb_conf*100)
        license_text = extract_text(image,boxes_np[ind])

        #print("Licence_number:  ",license_text)          # for  vehicle number  return 

        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.rectangle(image,(x,y-30),(x+w,y),(255,0,255),-1)
        cv2.rectangle(image,(x,y+h),(x+w,y+h+30),(0,0,0),-1) 


        cv2.putText(image,conf_text,(x,y-13),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        cv2.putText(image,license_text,(x,y+h+27),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    return image

# predictions 
def yolo_predictions(img,net):
    ## step-1: detections
    input_image, detections = get_detections(img,net)
    ## step-2: NMS
    boxes_np, confidences_np, index = non_maximum_supression(input_image, detections)
    ## step-3: Drawings
    result_img = drawings(img,boxes_np,confidences_np,index)
    cv2.imshow("webcam",result_img)
    return None


print("Starting...")
video_file = "last_video.mp4"
cap = cv2.VideoCapture(video_file)

pool = mp.Pool(processes=mp.cpu_count()-1)

start = time.time()
while True:
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.resize(frame,(600,400))
    pool.apply_async(yolo_predictions(frame,net))

pool.close()
pool.join()
cap.release()
cv2.destroyAllWindows()
print("Finished")
end = time.time()
print(end-start)