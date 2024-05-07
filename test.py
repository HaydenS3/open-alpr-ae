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
states = ['ALABAMA', 'ALASKA', 'ARIZONA', 'ARKANSAS', 'CALIFORNIA', 'COLORADO', 'CONNECTICUT', 'DELAWARE', 'FLORIDA', 'GEORGIA', 'HAWAII', 'IDAHO', 'ILLINOIS', 'INDIANA', 'IOWA', 'KANSAS', 'KENTUCKY', 'LOUISIANA', 'MAINE', 'MARYLAND', 'MASSACHUSETTS', 'MICHIGAN', 'MINNESOTA', 'MISSISSIPPI', 'MISSOURI', 'MONTANA', 'NEBRASKA', 'NEVADA', 'NEW HAMPSHIRE', 'NEW JERSEY', 'NEW MEXICO', 'NEW YORK', 'NORTH CAROLINA', 'NORTH DAKOTA', 'OHIO', 'OKLAHOMA', 'OREGON', 'PENNSYLVANIA', 'RHODE ISLAND', 'SOUTH CAROLINA', 'SOUTH DAKOTA', 'TENNESSEE', 'TEXAS', 'UTAH', 'VERMONT', 'VIRGINIA', 'WASHINGTON', 'WEST VIRGINIA', 'WISCONSIN', 'WYOMING','AMERICAN SAMOA', 'PUERTO RICO', 'U S VIRGIN ISLANDS']

tested = []
correct_blur = []
correct_noise = []
incorrect_blur = []
incorrect_noise = []
no_plate_blur = []
no_plate_noise = []
no_plate_original = []

for i in range(len(states)):
    directory = "dataset1/new_plates/train/" + states[i] + "/"
    d2 = directory + "noise/"
    d3 = directory + "blur/"

    tested.append(0)
    correct_blur.append(0)
    correct_noise.append(0)
    incorrect_blur.append(0)
    incorrect_noise.append(0)
    no_plate_blur.append(0)
    no_plate_noise.append(0)
    no_plate_original.append(0)


    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        print(filename)
        if not filename.endswith(".jpg"):
            continue
        original = run_openalpr(directory + filename)
        blur = run_openalpr(d3 + filename)
        noise = run_openalpr(d2 + filename)
        tested[i] += 1
        if original == "no plate":
            no_plate_original[i] += 1
        else:
            if blur == "no plate":
                no_plate_blur[i] += 1
            elif blur != original:
                incorrect_blur[i] += 1
            else:
                correct_blur[i] += 1
            if noise == "no plate":
                no_plate_noise[i] += 1
            elif noise != original:
                incorrect_noise[i] += 1
            else:
                correct_noise[i] += 1
        print("Original: {}, Blur: {}, Noise: {}".format(original,blur,noise))

#print stats
tested.append(0)
correct_blur.append(0)
correct_noise.append(0)
incorrect_blur.append(0)
incorrect_noise.append(0)
no_plate_blur.append(0)
no_plate_noise.append(0)
no_plate_original.append(0)
j = len(states)
states.append("Total")

for i in range(len(states)):
    print("State: {}. Read {}/{} original plates.".format(states[i], tested[i] - no_plate_original[i], tested[i]))
    print("State: {}. Succesfully read {}/{} blurred plates".format(states[i], correct_blur[i],tested[i] - no_plate_original[i]))
    print("State: {}. Succesfully read {}/{} noisy plates".format(states[i], correct_noise[i],tested[i] - no_plate_original[i]))

    tested[j] += tested[i]
    correct_blur[j] += correct_blur[i]
    correct_noise[j] += correct_noise[i]
    incorrect_blur[j] += incorrect_blur[i]
    incorrect_noise[j] += incorrect_noise[i]
    no_plate_blur[j] += no_plate_blur[i]
    no_plate_noise[j] += no_plate_noise[i]
    no_plate_original[j] += no_plate_original[i]

# Remember to close the OpenALPR instance
alpr.unload()
