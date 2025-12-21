import torch
import os
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

# =========================
# DEVICE
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD VIT HASIL FINETUNE
# =========================
CHECKPOINT_PATH = "vit/vit_simple/vit_out/checkpoint-1563"

processor = ViTImageProcessor.from_pretrained(CHECKPOINT_PATH)

model = ViTForImageClassification.from_pretrained(
    CHECKPOINT_PATH
).to(device)

model.eval()

# =========================
# PREDICT 1 IMAGE
# =========================
def predict_image(img_path):
    image = Image.open(img_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    conf, pred = torch.max(probs, dim=1)

    return pred.item(), conf.item()

# =========================
# EVALUATE SEMUA SR METHOD
# =========================
methods = ["edsr", "esrgan"]

for method in methods:
    folder = f"results/{method}"
    print(f"\n=== {method.upper()} ===")

    for img_name in sorted(os.listdir(folder)):
        img_path = os.path.join(folder, img_name)
        pred, conf = predict_image(img_path)

        print(f"{img_name} → class {pred}, confidence {conf:.3f}")
