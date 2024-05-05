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

# Define a simple model (replace this with your actual model)
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(224, 224, 1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))  # Resize image to match model input size
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = np.expand_dims(img, axis=2)  # Add channel dimension
    img = img.astype(np.float32)
    return img

def predict_with_ocr(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file {img_path} not found.")
    
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Unable to read image file {img_path}.")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB format
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        temp_filename = temp_file.name
        cv2.imwrite(temp_filename, img_rgb)
        text = pytesseract.image_to_string(temp_filename)
    os.unlink(temp_filename)  # Delete temporary file
    return text

def adversarial_attack(img_path, target_text, epochs=10):
    for epoch in range(epochs):
        img = preprocess_image(img_path)
        img_tensor = tf.convert_to_tensor(img)

        # Reshape the image tensor to match the model's input shape
        img_tensor = tf.expand_dims(img_tensor, axis=0)  # Add batch dimension
        img_tensor = tf.expand_dims(img_tensor, axis=-1)  # Add channel dimension

        # Get original prediction
        orig_text = predict_with_ocr(img_path)
        print("Original prediction:", orig_text)

        # Generate adversarial example using Fast Gradient Sign Method (FGSM)
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            prediction = model(img_tensor)
            loss = -tf.keras.losses.sparse_categorical_crossentropy([target_text], prediction)

        gradient = tape.gradient(loss, img_tensor)
        perturbation = 0.1 * np.sign(gradient)  # Increase perturbation magnitude
        adv_img = img_tensor + perturbation

        # Clip the perturbed image to [0, 255] range
        adv_img = tf.clip_by_value(adv_img, 0.0, 255.0)

        # Update model using adversarial example
        model.fit(adv_img, np.array([target_text]), verbose=0)

        adv_img = np.squeeze(adv_img.numpy(), axis=0)

        # Delete existing adversarial image if exists
        if os.path.exists("adversarial_image.jpg"):
            os.remove("adversarial_image.jpg")

        # Save adversarial image
        adv_img_rgb = cv2.cvtColor(adv_img, cv2.COLOR_GRAY2RGB)  # Convert to RGB format
        cv2.imwrite("adversarial_image.jpg", adv_img_rgb)

        # Get prediction on adversarial example
        adv_text = predict_with_ocr("adversarial_image.jpg")
        print("Adversarial prediction:", adv_text)

def test_adversarial_with_openalpr(adversarial_img_path):
    result = alpr.recognize_file(adversarial_img_path)
    if result['results']:
        plate = result['results'][0]['plate']
        confidence = result['results'][0]['confidence']
        print("License Plate:", plate)
        print("Confidence:", confidence)
    else:
        print("No license plate detected.")


# Example usage
#img_path = input('Enter path to file: ')
img_path = "dataset2/images/Cars16.png"
#target_text = input('Enter target license plate text: ')
target_text = 3
adversarial_attack(img_path, target_text, epochs=10)

# Test adversarial image with OpenALPR
test_adversarial_with_openalpr("adversarial_image.jpg")

# Remember to close the OpenALPR instance
alpr.unload()
