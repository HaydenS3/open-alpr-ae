import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import pytesseract
import openalpr
import cv2
import tempfile

# Load the OpenALPR model
alpr = openalpr.Alpr("us", "conf.txt", "runtime_data")
if not alpr.is_loaded():
    print("Error loading OpenALPR")
    exit(1)

def run_openalpr(adversarial_img_path):
    result = alpr.recognize_file(adversarial_img_path)
    if result['results']:
        plate = result['results'][0]['plate']
        confidence = result['results'][0]['confidence']
        return plate
    else:
        return "no plate"

# Perform the operation for all files in the dataset1/new_plates/train/MISSOURI directory
directory = "dataset1/new_plates/train/MISSOURI/"
d2 = "dataset1/new_plates/train/MISSOURI/noise/"
d3 = "dataset1/new_plates/train/MISSOURI/blur/"

tested = 0
correct_blur = 0
correct_noise = 0
incorrect_blur = 0
incorrect_noise = 0
no_plate_blur = 0
no_plate_noise = 0
no_plate_original = 0


for file in os.listdir(directory):
    filename = os.fsdecode(file)
    print(filename)
    if not filename.endswith(".jpg"):
        continue
    original = run_openalpr(directory + filename)
    blur = run_openalpr(d3 + filename)
    noise = run_openalpr(d2 + filename)
    tested += 1
    if original == "no plate":
        no_plate_original += 1
    else:
        if blur == "no plate":
            no_plate_blur += 1
        elif blur != original:
            incorrect_blur += 1
        else:
            correct_blur += 1
        if noise == "no plate":
            no_plate_noise += 1
        elif noise != original:
            incorrect_noise += 1
        else:
            correct_noise += 1
    print("Original: {}, Blur: {}, Noise: {}".format(original,blur,noise))

#print stats
print("Read {}/{} original plates.".format(tested - no_plate_original, tested))
print("Succesfully read {}/{} blurred plates".format(correct_blur,tested - no_plate_original))
print("Succesfully read {}/{} noisy plates".format(correct_noise,tested - no_plate_original))

# Remember to close the OpenALPR instance
alpr.unload()
