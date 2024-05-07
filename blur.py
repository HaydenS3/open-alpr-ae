import os
from PIL import Image, ImageFilter

def blur_jpg_images_in_directory(directory):
    # tweak blur: different filter mode, different radius
    blur_folder = os.path.join(directory, "blur")
    if not os.path.exists(blur_folder):
        os.makedirs(blur_folder)

    files = os.listdir(directory)

    jpg_files = [f for f in files if f.endswith('.jpg')]

    for jpg_file in jpg_files:
        jpg_path = os.path.join(directory, jpg_file)
        with Image.open(jpg_path) as img:
            blurred_img = img.filter(ImageFilter.GaussianBlur(radius=2))
            blurred_img.save(os.path.join(blur_folder, jpg_file))

if __name__ == "__main__":
    states = ['ALABAMA', 'ALASKA', 'ARIZONA', 'ARKANSAS', 'CALIFORNIA', 'COLORADO', 'CONNECTICUT', 'DELAWARE', 'FLORIDA', 'GEORGIA', 'HAWAII', 'IDAHO', 'ILLINOIS', 'INDIANA', 'IOWA', 'KANSAS', 'KENTUCKY', 'LOUISIANA', 'MAINE', 'MARYLAND', 'MASSACHUSETTS', 'MICHIGAN', 'MINNESOTA', 'MISSISSIPPI', 'MISSOURI', 'MONTANA', 'NEBRASKA', 'NEVADA', 'NEW HAMPSHIRE', 'NEW JERSEY', 'NEW MEXICO', 'NEW YORK', 'NORTH CAROLINA', 'NORTH DAKOTA', 'OHIO', 'OKLAHOMA', 'OREGON', 'PENNSYLVANIA', 'RHODE ISLAND', 'SOUTH CAROLINA', 'SOUTH DAKOTA', 'TENNESSEE', 'TEXAS', 'UTAH', 'VERMONT', 'VIRGINIA', 'WASHINGTON', 'WEST VIRGINIA', 'WISCONSIN', 'WYOMING','AMERICAN SAMOA', 'PUERTO RICO', 'U S VIRGIN ISLANDS']
    current_directory = "dataset1/new_plates/train/"
    for i in range(len(states)):
        current_directory = "dataset1/new_plates/train/" + states[i] + "/"
        blur_jpg_images_in_directory(current_directory)
    print("Blurring completed")
