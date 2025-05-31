import os
from PIL import Image
import numpy as np

dir = "/home/leprieto/tfm/resources/dataset/casting_data"

greyscale = True

def is_greyscale(image_path):
    image = Image.open(image_path)
    image = image.convert('RGB')
    img_array = np.array(image)
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    if np.array_equal(r, g) and np.array_equal(g, b):
        return True
    else:
        return False
                
for root1, dirs1, files1 in os.walk(dir):
        for file1 in files1:
            greyscale &= is_greyscale(os.path.join(root1, file1))

if is_greyscale:
    print("All images are greyscale.")
else: 
    print("Non greyscale images found.")
