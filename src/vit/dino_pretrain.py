import os
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from transformers import ViTModel

from utils_dataset import UnlabeledImageDataset

# -----------------------------
# Transform
# -----------------------------
transform = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ToTensor()
])

# dataset = UnlabeledImageDataset(
#     root_dir="data",   # <- SYMLINK
#     transform=transform
# )
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'LSDIR')
dataset = UnlabeledImageDataset(root)

loader = DataLoader(dataset, batch_size=16, shuffle=True)

# -----------------------------
# DINO Head
# -----------------------------
class DINOHead(nn.Module):
    def __init__(self, in_dim=768, out_dim=65536):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, out_dim)
        )

    def forward(self, x):
        return self.mlp(x)

student = ViTModel.from_pretrained("google/vit-base-patch16-224")
teacher = ViTModel.from_pretrained("google/vit-base-patch16-224")

student_head = DINOHead()
teacher_head = DINOHead()

teacher.load_state_dict(student.state_dict())
teacher_head.load_state_dict(student_head.state_dict())

for p in teacher.parameters():
    p.requires_grad = False

optimizer = torch.optim.Adam(
    list(student.parameters()) + list(student_head.parameters()),
    lr=3e-4
)

criterion = nn.CrossEntropyLoss()

# -----------------------------
# Training
# -----------------------------
for epoch in range(10):
    for imgs in loader:
        s_feat = student(imgs).last_hidden_state[:, 0]
        t_feat = teacher(imgs).last_hidden_state[:, 0]

        s_out = student_head(s_feat)
        t_out = teacher_head(t_feat)

        loss = criterion(s_out, t_out.argmax(dim=1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} | Loss: {loss.item():.2f}")

torch.save(student.state_dict(), "vit_dino_pretrained.pth")
