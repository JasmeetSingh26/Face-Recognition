import cv2
import os

# Path to the folder containing images
image_folder_path = 'C:/Users/Acer/Desktop/mini project/myenv/OpenCV-Face-Recognition/Brad Pitt'

# Initialize the face detector
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

id = input("Enter Your ID: ")
count = 0

# Loop through all files in the image folder
for filename in os.listdir(image_folder_path):
        img_path = os.path.join(image_folder_path, filename)
        frame = cv2.imread(img_path)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            count += 1
            cv2.imwrite('C:/Users/Acer/Desktop/mini project/myenv/OpenCV-Face-Recognition/datasets/User.'+str(id)+"."+str(count)+".jpg", gray[y:y+h, x:x+w])
            
        


print("Dataset Collection Done..................")
