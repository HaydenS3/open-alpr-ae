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
    current_directory = os.getcwd()
    blur_jpg_images_in_directory(current_directory)
    print("Blurring completed")