import os
from PIL import Image

lr_dir = "C:/Users/natha/OneDrive/Desktop/Documents/CIT/Semester 7/Computer Vision/Dataset_UAS_ready/train/LR"
hr_dir = "C:/Users/natha/OneDrive/Desktop/Documents/CIT/Semester 7/Computer Vision/Dataset_UAS_ready/train/HR"

name = os.listdir(lr_dir)[0]

lr = Image.open(os.path.join(lr_dir, name))
hr = Image.open(os.path.join(hr_dir, name))

print("LR size:", lr.size)
print("HR size:", hr.size)

