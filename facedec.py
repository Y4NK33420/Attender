import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
ptime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()


while 1:
    _,img = cap.read()
    imgrgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgrgb)
    #print(results)
    if results.detections:
        for id,detection in enumerate(results.detections):
            
            bb = detection.location_data.relative_bounding_box
            print(id,detection,bb)
            ih,iw,ic = img.shape
            bbox = int(bb.xmin * iw), int(bb.ymin*ih),\
                int(bb.width*iw), int(bb.height*ih)
            cv2.rectangle(img,bbox,(255,0,0),2)
            mpDraw.draw_detection(img,detection)


    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    cv2.putText(img,f'FPS {int(fps)}',(20,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),3)
    cv2.imshow('image',img)
    cv2.waitKey(1)