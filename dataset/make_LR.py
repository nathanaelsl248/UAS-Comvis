import os
from PIL import Image

def make_lr(src_dir, dst_dir, scale=4):
    os.makedirs(dst_dir, exist_ok=True)

    for img_name in os.listdir(src_dir):
        img_path = os.path.join(src_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        w, h = img.size
        lr = img.resize((w // scale, h // scale), Image.BICUBIC)
        lr.save(os.path.join(dst_dir, img_name))

# TRAIN
make_lr(
    "C:/Users/natha/OneDrive/Desktop/Documents/CIT/Semester 7/Computer Vision/Dataset_UAS_ready/train/HR",
    "C:/Users/natha/OneDrive/Desktop/Documents/CIT/Semester 7/Computer Vision/Dataset_UAS_ready/train/LR"
)

# VAL
make_lr(
    "C:/Users/natha/OneDrive/Desktop/Documents/CIT/Semester 7/Computer Vision/Dataset_UAS_ready/val/HR",
    "C:/Users/natha/OneDrive/Desktop/Documents/CIT/Semester 7/Computer Vision/Dataset_UAS_ready/val/LR"
)

print("LR generation done")
