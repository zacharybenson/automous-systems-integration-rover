import os
from PIL import Image

# The purpose of this file is to remove corrupted files from a directoy
img_dir = "/Users/zacharybenson/Desktop/w/"
files = [file for file in os.listdir(img_dir) if file.endswith("_w.png")]

for filename in files:
    filepath = os.path.join(img_dir, filename)
    try:
        Image.open(filepath)
    except Exception as e:
        print("removing")
        os.remove(filepath)
