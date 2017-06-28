import cv2
import os

cascade_file = "../get_data/lbpcascade_animeface.xml"

def detect_face(image, classifier=None):

    if classifier == None:
        if not os.path.isfile(cascade_file):
            raise RuntimeError("Cascade file not found")
        classifier = cv2.CascadeClassifier(cascade_file)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)

    return classifier.detectMultiScale(image, scaleFactor = 1.1, minNeighbors = 5, minSize = (24,24))

def crop_image(image, face_coords):
    x, y, w, h = face_coords

    face = image[y:(y+h), x:(x+w)]
    return face

