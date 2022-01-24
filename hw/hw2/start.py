import torch
import os
from PIL import Image

base_image_path = "images"
image1_name = "img1.jpg"
image2_name = "img2.jpg"

img1 = Image.open(os.path.join(base_image_path,image1_name))
img2 = Image.open(os.path.join(base_image_path,image2_name))

img1.show()
