import cv2
import numpy as np
import os
import streamlit as st
from PIL import Image



# Function to capture images for dataset collection
def capture_dataset(id, count):
    video = cv2.VideoCapture(0)
    facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            count =count+ 1
            cv2.imwrite(f'datasets/User.{id}.{count}.jpg', gray[y:y+h, x:x+w])
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

        cv2.imshow("Frame", frame)

        k = cv2.waitKey(1)

        if count > 500 :
            break

    video.release()
    cv2.destroyAllWindows()
    print("Dataset Collection Done..................")

# Function to train the recognizer
# Function to train the recognizer
def train_recognizer(path):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("C:/Users/Acer/Desktop/mini project/myenv/OpenCV-Face-Recognition/FacialRecognition/haarcascade_frontalface_default.xml")

    def getImagesAndLabels(path):
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
        faceSamples=[]
        ids = []

        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L') 
            img_numpy = np.array(PIL_img,'uint8')

            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)

            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)

        return faceSamples, ids

    st.write("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces, ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))

    recognizer.write('C:/Users/Acer/Desktop/mini project/myenv/OpenCV-Face-Recognition/trainer/trainer1.yml') 
    st.write("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))


# Function to recognize faces and mark attendance
def mark_attendance():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('C:/Users/Acer/Desktop/mini project/myenv/OpenCV-Face-Recognition/trainer/trainer1.yml')
    cascadePath = "C:/Users/Acer/Desktop/mini project/myenv/OpenCV-Face-Recognition/FacialRecognition/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)

    font = cv2.FONT_HERSHEY_SIMPLEX
    names = {0: 'None', 1: 'Jasmeet', 2: 'Tom', 3: 'Brad', 4: 'User'} 

    cam = cv2.VideoCapture(0)
    cam.set(3, 640) 
    cam.set(4, 480) 

    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    marked_attendees = []
    unmarked_attendees = ['None',  'Jasmeet',  'Tom',  'Brad',  'User']


    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(int(minW), int(minH)))

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            if confidence < 50:
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
                marked_attendees.append(id)
                if id in unmarked_attendees:
                    unmarked_attendees.remove(id)
            else:
                id = "Unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        cv2.imshow('camera', img)

        cv2.waitKey(1)
        

        k = cv2.waitKey(1)
        if k % 256 == 27:  
            # Press 'ESC' to quit
            print("\n [INFO] Exiting Program and cleanup stuff")
            break

    cam.release()
    cv2.destroyAllWindows()

    st.write("### Marked Attendees")
    st.write(list(set(marked_attendees)))
    st.write("### Unmarked Attendees")
    st.write(list(set(unmarked_attendees)))



# Streamlit UI
def main():
    st.title("Attendance System")

    task = st.sidebar.selectbox(
        "Choose a task",
        ["Capture Dataset", "Train Recognizer", "Mark Attendance"]
    )

    if task == "Capture Dataset":
        st.write("### Capture Dataset")
        id = st.text_input("Enter Your ID:")
        count = 0
        if st.button("Start Capturing"):
            capture_dataset(id, count)

    elif task == "Train Recognizer":
        st.write("### Train Recognizer")
        if st.button("Train"):
            train_recognizer("datasets/")

    elif task == "Mark Attendance":
        st.write("### Mark Attendance")
        if st.button("Start Marking"):
            mark_attendance()

if __name__ == "__main__":
    main()
