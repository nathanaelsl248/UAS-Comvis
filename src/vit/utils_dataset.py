import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")

class UnlabeledImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.images = []

        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith(IMG_EXTENSIONS):
                    self.images.append(os.path.join(root, f))

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

        print(f"[Dataset] Found {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        return self.transform(img)
