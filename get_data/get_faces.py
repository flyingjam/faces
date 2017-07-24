import cv2
import os
#note,
#conda install -c conda-forge opencv=3.2.0 
#for using conda on python 3.6

cascade_file = "./get_data/lbpcascade_animeface.xml"

def detect_face(image, classifier=None, scaleFactor = 1.2, minNeighbors = 4, minSize = (24,24)):

    if classifier == None:
        if not os.path.isfile(cascade_file):
            raise RuntimeError("Cascade file not found")
        classifier = cv2.CascadeClassifier(cascade_file)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)

    return classifier.detectMultiScale(image,
                                       scaleFactor = scaleFactor,
                                       minNeighbors = minNeighbors,
                                       minSize = minSize)

def create_classifier(path = cascade_file):
    if not os.path.isfile(path):
        raise RuntimeError("Cascade file not found")
    return cv2.CascadeClassifier(path)

def crop_image(image, face_coords, size=(32,32), interpolation = cv2.INTER_AREA):
    x, y, w, h = face_coords

    face = image[y:(y+h), x:(x+w)]
    resized_face = cv2.resize(face, size, interpolation=interpolation)
    return resized_face

