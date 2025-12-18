import torch
import numpy as np
from transformers import ViTModel
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans

from utils_dataset import UnlabeledImageDataset

dataset = UnlabeledImageDataset("data/LSDIR")
loader = DataLoader(dataset, batch_size=16)

model = ViTModel.from_pretrained("google/vit-base-patch16-224")
model.load_state_dict(torch.load("vit_dino_pretrained.pth"))
model.eval()

features = []

with torch.no_grad():
    for imgs in loader:
        feat = model(imgs).last_hidden_state[:, 0]
        features.append(feat.numpy())

features = np.vstack(features)

kmeans = KMeans(n_clusters=5, random_state=42)
pseudo_labels = kmeans.fit_predict(features)

np.save("pseudo_labels.npy", pseudo_labels)
