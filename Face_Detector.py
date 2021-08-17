import cv2
from random import randrange

trained_face_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# img1=cv2.imread('RDJ.jpg')

webcam = cv2.VideoCapture(0)

while True:

    succesful_frame_read, frame=webcam.read()

    grey_scale_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    face_coordinates=trained_face_data.detectMultiScale(grey_scale_img)

    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),3)

    cv2.imshow('WebCame Detector',frame)
    key=cv2.waitKey(1)

    if key==81 or key==113:
        break

webcam.release()
     

print("Code Completed")

