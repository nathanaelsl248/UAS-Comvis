import os
import shutil
import random

src = "C:/Users/natha/OneDrive/Desktop/Documents/CIT/Semester 7/Computer Vision/Dataset_UAS_ready/all_HR"
train_dst = "C:/Users/natha/OneDrive/Desktop/Documents/CIT/Semester 7/Computer Vision/Dataset_UAS_ready/train/HR"
val_dst = "C:/Users/natha/OneDrive/Desktop/Documents/CIT/Semester 7/Computer Vision/Dataset_UAS_ready/val/HR"

os.makedirs(train_dst, exist_ok=True)
os.makedirs(val_dst, exist_ok=True)

images = os.listdir(src)
random.shuffle(images)

split = int(0.8 * len(images))

for img in images[:split]:
    shutil.copy(os.path.join(src, img), os.path.join(train_dst, img))

for img in images[split:]:
    shutil.copy(os.path.join(src, img), os.path.join(val_dst, img))

print("Train images:", split)
print("Val images:", len(images) - split)
