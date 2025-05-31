import os
import shutil
dir = "/home/leprieto/tfm/resources/dataset/casting_data" 

for root, dirs, files in os.walk(dir):
    for file in files:
        vector = root.split("/")[-3:-1]
        uid_prefix = "_".join(vector) + "_"
        shutil.move(os.path.join(root, file), os.path.join(root, uid_prefix + file))