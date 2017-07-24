import os
import cv2
from subprocess import run
import get_data.get_faces as get_faces

def download_raw_dataset(tag, n_images = 1000, directory = "data/raw", download_url="https://danbooru.donmai.us/posts?tags={}"):
    arg_images = ["--images", "{}".format(n_images)]
    arg_url = [download_url.format(tag)]
    arg_directory = ["-d", "{}".format(directory)]
    args = ["gallery-dl"] + arg_images + arg_directory + arg_url
    return run(args, shell=True, check=True)

def convert_to_faces(tag, in_directory="data/raw/danbooru", out_directory = "data/faces/danbooru"):
    full_in = os.path.join(in_directory, tag)
    full_out = os.path.join(out_directory, tag)

    if not os.path.exists(full_out):
        os.makedirs(full_out)

    image_ends = (".png", ".PNG", ".jpg", ".jpeg", ".JPEG", ".JPG")
    classifier = get_faces.create_classifier()
    for filename in os.listdir(full_in):
        if filename.endswith(image_ends):
            filepath = os.path.join(full_in, filename)
            image = cv2.imread(filepath)
            faces = get_faces.detect_face(image, classifier)
            print(faces)
            for i, face in enumerate(faces):
                face_image = get_faces.crop_image(image, face)
                face_filename = str(i) + filename
                face_filepath = os.path.join(full_out, face_filename)
                cv2.imwrite(face_filepath, face_image)
                print("Saved face {} of {}".format(i, filename))


