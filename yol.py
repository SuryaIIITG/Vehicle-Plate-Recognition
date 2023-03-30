import matplotlib.pyplot as plt
import cv2
import numpy as np 
import os
import pytesseract as pt  
import plotly.express as px
import matplotlib.pyplot as plt 
import numpy as np
import csv
from datetime import datetime
import time
import pandas as pd
from timeline import car_time


# settings
INPUT_WIDTH =  640
INPUT_HEIGHT = 640
"""
# LOAD THE IMAGE 
img = cv2.imread('/home/pi/IOCL/my_custom_onnx/test/images/N8.jpeg')
#plt.imshow(img)     
#plt.show()  

cv2.namedWindow('test image',cv2.WINDOW_KEEPRATIO)
cv2.imshow('test image',img)
cv2.waitKey()
cv2.destroyAllWindows()
"""



# LOAD YOLO MODEL
net = cv2.dnn.readNetFromONNX('/home/pi/IOCL/best.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
print(net)




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

        print("Licence_number:  ",license_text)          # for  vehicle number  return 


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
        {} 
        print("Text of plate is: ",text)  
        
        return text
# predictions flow with return result
def yolo_predictions(img,net): 
    # step-1: detections
    input_image, detections = get_detections(img,net)
    # step-2: NMS
    boxes_np, confidences_np, index = non_maximum_supression(input_image, detections)
    # step-3: Drawings
    result_img = drawings(img,boxes_np,confidences_np,index)
    return result_img
# extrating text
def extract_text(image,bbox):
    x,y,w,h = bbox
    roi = image[y:y+h, x:x+w]
    print("Region",roi)
    
    if 0 in roi.shape:
        return 'no number'
    
    else:
        text = pt.image_to_string(roi)
        text = text.strip()
         # create Dataframe 
        raw_data={'date':[time.asctime(time.localtime(time.time()))],'vehicle_number':[text]} 
        df=pd.DataFrame(raw_data,columns=['date','vehicle_number'])
        df.to_csv('data.csv')

   
        #df=car_time()
        #df.to_csv('data.csv') 
        """
         #Data stored in csv file
        new=[]
        for i in text:   
            raw_data={'date':[time.asctime(time.localtime(time.time()))],'vehicle_number':[text]} 
            #raw_data =raw_data[['date','vehicle_number']]   
            df=pd.DataFrame(raw_data,columns=['date','vehicle_number']) 
            df.to_csv('data5.csv')
            """

            #print("Datatypes of Dataframe" , df.dtypes)
            #data_f=new.append(df)  
            #data_f.to_csv('data4.csv', header=False, index=False)            
            
        #with open('my_file.csv', 'w') as out:
        #    out.write(text) 

        # stored into csv file 
        #new_text=text.to_csv('plate.csv') 
        #print(new_text)
        #text1 = net(text)
        #store=text1.pandas().xyxy[0]
        #print(store)
        #np.unique(store[])

        print("Text of plate is:  ",text)
        
        return text

    
# test
img = cv2.imread('/home/pi/IOCL/my_custom_onnx/test/images/N8.jpeg')

results = yolo_predictions(img,net) 




# For webcam

cap = cv2.VideoCapture('/home/pi/IOCL/last_video.mp4') 
#cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()

    if ret == False:
        print('Unable to read video') 
        break

    results = yolo_predictions(frame,net)

    print("Passing video result ",results)

    cv2.namedWindow('YOLO',cv2.WINDOW_KEEPRATIO)
    cv2.imshow('YOLO',results)
    if cv2.waitKey(30) == 27 :
        break

cv2.destroyAllWindows()
cap.release()