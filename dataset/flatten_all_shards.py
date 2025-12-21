import os
import shutil

src_root = "C:/Users/natha/OneDrive/Desktop/Documents/CIT/Semester 7/Computer Vision/Dataset UAS"
dst_root = "C:/Users/natha/OneDrive/Desktop/Documents/CIT/Semester 7/Computer Vision/Dataset_UAS_ready/all_HR"


os.makedirs(dst_root, exist_ok=True)

count = 0

for shard in ["shard-00", "shard-01"]:
    shard_path = os.path.join(src_root, shard)

    for folder in os.listdir(shard_path):
        folder_path = os.path.join(shard_path, folder)

        if os.path.isdir(folder_path):
            for img in os.listdir(folder_path):
                src = os.path.join(folder_path, img)
                dst = os.path.join(dst_root, f"{count:07d}.png")
                shutil.copy(src, dst)
                count += 1

print("Total HR images:", count)
