import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import pytesseract
import openalpr
import cv2

# Load the OpenALPR model
alpr = openalpr.Alpr("us", "conf.txt", "runtime_data")
if not alpr.is_loaded():
    print("Error loading OpenALPR")
    exit(1)

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))  # Resize image to match model input size
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = np.expand_dims(img, axis=2)  # Add channel dimension
    img = img.astype(np.float32)
    return img

def predict_with_ocr(img_path):
    img = cv2.imread(img_path)
    text = pytesseract.image_to_string(img)
    return text

def adversarial_attack(img_path, target_text):
    img = preprocess_image(img_path)

    # Get original prediction
    orig_text = predict_with_ocr(img_path)
    print("Original prediction:", orig_text)

    # Generate adversarial example by adding noise
    adv_img = img + np.random.normal(scale=10, size=img.shape)

    # Save adversarial image
    cv2.imwrite("adversarial_image.jpg", adv_img)

    # Get prediction on adversarial example
    adv_text = predict_with_ocr("adversarial_image.jpg")
    hex_value = ''.join([format(ord(char), 'x') for char in adv_text])
    print("Adversarial prediction:", hex_value)

# Example usage
img_path = input('Enter path to file: ')
target_text = input('Enter target license plate text: ')
adversarial_attack(img_path, target_text)

# Remember to close the OpenALPR instance
alpr.unload()
