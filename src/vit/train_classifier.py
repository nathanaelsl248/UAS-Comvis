import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import ViTModel

from utils_dataset import UnlabeledImageDataset

class PseudoLabelDataset(Dataset):
    def __init__(self, base_dataset, labels):
        self.base = base_dataset
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.base[idx]
        return img, self.labels[idx]

base_dataset = UnlabeledImageDataset("data")
labels = torch.tensor(np.load("pseudo_labels.npy"))

dataset = PseudoLabelDataset(base_dataset, labels)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
classifier = nn.Linear(768, 5)

optimizer = torch.optim.Adam(
    list(vit.parameters()) + list(classifier.parameters()),
    lr=1e-4
)

criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    for imgs, y in loader:
        feat = vit(imgs).last_hidden_state[:, 0]
        logits = classifier(feat)

        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} | Loss: {loss.item():.2f}")