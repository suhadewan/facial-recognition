import os
import face_recognition as fr
import cv2
import face_recognition
import numpy as np
from time import sleep


def get_faces():
    encoded = {}

    for dirpath, dnames, fnames in os.walk("./faces"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file("faces/" + f)
                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding

    return encoded


## label images
def classify_face(im):
    
    faces = get_faces()
    faces_encoded = list(faces.values())
    known = list(faces.keys())
    img = cv2.imread(im, 1)
 
    face_locations = face_recognition.face_locations(img)
    unknown = face_recognition.face_encodings(img, face_locations)
    names = []
    ##unknown
    for face_encoding in unknown:
        matches = face_recognition.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"

        dist = face_recognition.face_distance(faces_encoded, face_encoding)
        idx = np.argmin(dist)
        if matches[idx]:
            name = known[idx]

        names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, names):
            # box
            cv2.rectangle(img, (left-20, top-20), (right+20, bottom+20), (255, 0, 0), 2)

            # printing name
            cv2.rectangle(img, (left-20, bottom -15), (right+20, bottom+20), (255, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name, (left -20, bottom + 15), font, 1.0, (255, 255, 255), 2)


    while True:

        cv2.imshow('Video', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return names 


print(classify_face("recognize.jpg"))


