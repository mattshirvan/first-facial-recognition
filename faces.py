import numpy as np
import cv2
import pickle

# haar cascade 
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')

# initialize recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trained.yml")

# load pickle file with training labels
labels = {}
with open("labels.pickle", "rb") as file:
    orig_labels = pickle.load(file)
    labels = {v:k for k, v in orig_labels.items()}

# capture object
cap = cv2.VideoCapture(0)

# iterate object
while True:
    # capture frames
    ret, frame = cap.read()

    # convert to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    # iterate over grayed faces
    for (x, y, w, h) in faces:    
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # deep learning recognizer
        id_, conf = recognizer.predict((roi_gray))
        if conf >= 45 and conf <= 85:            
            # create label identifier
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)

        # create a png file
        img_item = "my-image.png"
        cv2.imwrite(img_item, roi_gray)

        # add rectangle
        color = (255, 0, 0)
        stroke = 2
        x_coord = x + w
        y_coord = y + h
        cv2.rectangle(frame, (x, y), (x_coord, y_coord), color , stroke)

    # display frame result in color
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# release captured frame
cap.release()
cv2.destroyAllWindows()
