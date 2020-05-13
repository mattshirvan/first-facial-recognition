import os 
from PIL import Image
import numpy as np
import cv2
import pickle

# set base directory path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# set image directory path
image_dir = os.path.join(BASE_DIR, "images")

# haar cascade 
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')

# create recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# training label ids
current_id = 0
label_ids = {}

# training labels 
y_labels = []
x_train = []

# walk through directories
for root, dirs, files, in os.walk(image_dir):
    # iterate over files
    for file in files:
        # if the files format is an image
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path))            

            # issue label ids to keep track
            if not label in label_ids:                
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]

            # convert to grayscale and numpy array
            pil_image = Image.open(path).convert("L")
            size = (550, 550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(final_image, "uint8")
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

            # iterate through faces append training data
            for (x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

# create pickle object
with open("labels.pickle", "wb") as files:
    pickle.dump(label_ids, files)

# save recognizer as yaml
recognizer.train(x_train, np.array(y_labels))
recognizer.save("trained.yml")
