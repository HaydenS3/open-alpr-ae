import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import openalpr

# Load the OpenALPR model
alpr = openalpr.Alpr("us", "conf.txt", "runtime_data")
if not alpr.is_loaded():
    print("Error loading OpenALPR")
    exit(1)

# Load ResNet50 model for crafting adversarial examples
model = ResNet50(weights='imagenet')

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def predict_with_openalpr(img_path):
    results = alpr.recognize_file(img_path)
    if results['results']:
        plate = results['results'][0]['plate']
        return plate
    else:
        return None

def adversarial_attack(img_path, target_label):
    img = preprocess_image(img_path)
    orig_pred = model.predict(img)
    orig_label = decode_predictions(orig_pred, top=1)[0][0][1]
    print("Original prediction:", orig_label)

    target = tf.one_hot(target_label, 1000)
    target = tf.reshape(target, (1, 1000))

    with tf.GradientTape() as tape:
        tape.watch(img)
        prediction = model(img)
        loss = tf.keras.losses.categorical_crossentropy(target, prediction)

    gradient = tape.gradient(loss, img)
    perturbation = 0.01 * tf.sign(gradient)
    adv_img = img + perturbation
    adv_pred = model.predict(adv_img)
    adv_label = decode_predictions(adv_pred, top=1)[0][0][1]
    print("Adversarial prediction:", adv_label)

# Example usage
img_path = input('Enter path to file: ')
target_label = 22  # Change this to the target class you want
adversarial_attack(img_path, target_label)

# Remember to close the OpenALPR instance
alpr.unload()

