import os
from PIL import Image

def convert_jpg_to_png(src_dir, dst_dir):
    for dir_ in os.listdir(src_dir):
        os.makedirs(os.path.join(dst_dir, dir_), exist_ok=True)
    
        files = os.listdir(os.path.join(src_dir, dir_))
        for file in files:
            # Check if the file is a JPG image
            if file.lower().endswith(".jpg"):
                # Open the image using PIL
                image = Image.open(os.path.join(src_dir, dir_, file))
                
                # Get the file name without the extension
                file_name = os.path.splitext(file)[0]
                
                # Save the PNG image to the destination directory
                image.save(os.path.join(dst_dir, f"{dir_}/{file_name}.png"))
                
                # Close the image
                image.close()

# Usage example
src_dir = "./data/7PC/images"
dst_dir = "./data/Derm7pt/images"
convert_jpg_to_png(src_dir, dst_dir)