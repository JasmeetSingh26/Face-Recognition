import os
import cv2

# Input and output directories
input_folder = 'C:/Users/Acer/Desktop/mini project/myenv/OpenCV-Face-Recognition/Brad Pitt/'
output_folder = 'C:/Users/Acer/Desktop/mini project/myenv/OpenCV-Face-Recognition/datasets/'

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

id = input("Enter User ID: ")
count = 0


for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        filepath = os.path.join(input_folder, filename)
        frame = cv2.imread(filepath)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            count += 1
  
            cv2.imwrite(os.path.join(output_folder, f'User.{id}.{count}.jpg'), gray[y:y+h, x:x+w])

print("Dataset Collection Done.")
