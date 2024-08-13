import cv2
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('C:/Users/Acer/Desktop/mini project/myenv/OpenCV-Face-Recognition/trainer/trainer1.yml')
cascadePath = "C:/Users/Acer/Desktop/mini project/myenv/OpenCV-Face-Recognition/FacialRecognition/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX
names = {0: 'None', 1: 'Jasmeet', 2: 'Tom', 3: 'Brad ', 4: '', 5: 'Anshul'} 

cam = cv2.VideoCapture(0)




while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        if confidence < 45:
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
    
        else:
            id = "Unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    cv2.imshow('camera', img)

    k = cv2.waitKey(1)
    if k == 27:  
        # Press 'ESC' to quit
        print("\n Exiting Program and cleanup stuff")
        break

cam.release()
cv2.destroyAllWindows()
