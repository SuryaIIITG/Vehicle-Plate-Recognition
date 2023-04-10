import matplotlib.pyplot as plt
import cv2
import numpy as np 
import pytesseract as pt  
import numpy as np
from datetime import datetime
import time
#from timeline import car_time
import pandas as pd


# settings
INPUT_WIDTH =  640
INPUT_HEIGHT = 640

# LOAD YOLO MODEL
net = cv2.dnn.readNetFromONNX('best1.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
print("Model Executed" ,net) 


# All function 
def get_detections(img,net):
    # CONVERT IMAGE TO YOLO FORMAT
    image = img.copy()
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
    return result_img 


def extract_text(image,bbox):
    x,y,w,h = bbox 
    roi = image[y:y+h, x:x+w]
    
    if 0 in roi.shape: 
        return ''
    
    else:
        text = pt.image_to_string(roi)
        text = text.strip()
        print("Text of plate is: ",text)  
        
        return text
    
    
                 # This is testing image number from images only 

#img = cv2.imread('/home/dki/vehicle_plate_recognition-main/my_custom_onnx-20230409T135443Z-001/my_custom_onnx/test/images/N45.jpeg')
#img = cv2.imread('/home/dki/vehicle_plate_recognition-main/my_custom_onnx-20230409T135443Z-001/my_custom_onnx/test/images/N78.jpeg')
#img = cv2.imread('/home/dki/vehicle_plate_recognition-main/my_custom_onnx-20230409T135443Z-001/my_custom_onnx/test/images/N92.jpeg')
#results = yolo_predictions(img,net) 



             # For webcam 

cap = cv2.VideoCapture('last_video.mp4') 
#cap =cv2.cvtColor(cap,cv2.COLOR_BGR2RGB)
#cap = cv2.VideoCapture(0) 

# used to record the time when we processed last frame
prev_frame_time = 0

# used to record the time at which we processed current frame
new_frame_time = 0

while True:
    ret, frame = cap.read()

    if ret == False:
        print('Unable to read video') 
        break

    	# Our operations on the frame come here
	#gray = frame
        
    new_frame_time = time.time()  
    
    font = cv2.FONT_HERSHEY_SIMPLEX 
	# time when we finish processing for this frame
    new_frame_time = time.time() 

	# Calculating the fps

	# fps will be number of frame processed in given time frame
	# since their will be most of time error of 0.001 second
	# we will be subtracting it to get more accurate result
    fps = 1/(new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

	# converting the fps into integer
    fps = int(fps)

	# converting the fps to string so that we can display it on frame
	# by using putText function
    fps = str(fps)

	# putting the FPS count on the frame
    cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

    results = yolo_predictions(frame,net) 

    #print("Passing video result ",results) 

    cv2.namedWindow('YOLO',cv2.WINDOW_KEEPRATIO)  
    cv2.imshow('YOLO',results) 
    if cv2.waitKey(1) == 27 : 
        break

cv2.destroyAllWindows() 
cap.release() 
