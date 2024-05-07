import os
import matplotlib.pyplot as plt
from skimage import io, util

def apply_random_noise(image_path, mode='gaussian'):
    # tweak random noise options: mode, mean, amount, salt, pepper
    image = io.imread(image_path)
    noisy_image = util.random_noise(image, mode=mode)
    return noisy_image

def process_images_in_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg"):
            input_image_path = os.path.join(input_dir, filename)
            noisy_image = apply_random_noise(input_image_path)
            output_image_path = os.path.join(output_dir, filename)
            io.imsave(output_image_path, (noisy_image * 255).astype('uint8'))

def main():
    states = ['ALABAMA', 'ALASKA', 'ARIZONA', 'ARKANSAS', 'CALIFORNIA', 'COLORADO', 'CONNECTICUT', 'DELAWARE', 'FLORIDA', 'GEORGIA', 'HAWAII', 'IDAHO', 'ILLINOIS', 'INDIANA', 'IOWA', 'KANSAS', 'KENTUCKY', 'LOUISIANA', 'MAINE', 'MARYLAND', 'MASSACHUSETTS', 'MICHIGAN', 'MINNESOTA', 'MISSISSIPPI', 'MISSOURI', 'MONTANA', 'NEBRASKA', 'NEVADA', 'NEW HAMPSHIRE', 'NEW JERSEY', 'NEW MEXICO', 'NEW YORK', 'NORTH CAROLINA', 'NORTH DAKOTA', 'OHIO', 'OKLAHOMA', 'OREGON', 'PENNSYLVANIA', 'RHODE ISLAND', 'SOUTH CAROLINA', 'SOUTH DAKOTA', 'TENNESSEE', 'TEXAS', 'UTAH', 'VERMONT', 'VIRGINIA', 'WASHINGTON', 'WEST VIRGINIA', 'WISCONSIN', 'WYOMING','AMERICAN SAMOA', 'PUERTO RICO', 'U S VIRGIN ISLANDS']
    for i in range(len(states)):
        input_dir = "dataset1/new_plates/train/" + states[i] + "/"
        output_dir = os.path.join(input_dir, "noise")
        process_images_in_directory(input_dir, output_dir)

if __name__ == "__main__":
    main()
