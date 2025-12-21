import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer
)

# =====================
# DEVICE
# =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# =====================
# DATASET
# =====================
dataset = load_dataset("cifar10")

labels = dataset["train"].features["label"].names
num_labels = len(labels)

# =====================
# IMAGE PROCESSOR
# =====================
processor = ViTImageProcessor.from_pretrained(
    "google/vit-base-patch16-224-in21k"
)

# =====================
# PREPROCESS (BENAR)
# =====================
def preprocess(examples):
    inputs = processor(
        images=examples["img"],
        return_tensors="pt"
    )
    inputs["labels"] = examples["label"]
    return inputs

dataset = dataset.map(
    preprocess,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# =====================
# MODEL
# =====================
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=num_labels,
    ignore_mismatched_sizes=True
).to(device)

# =====================
# METRICS
# =====================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": (preds == labels).mean()}

# =====================
# TRAINING ARGS
# =====================
args = TrainingArguments(
    output_dir="./vit_out",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=1,   # ⏱️ cepat
    learning_rate=2e-4,
    fp16=True,
    logging_steps=50,
    report_to="none"
)

# =====================
# TRAINER
# =====================
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
)

# =====================
# RUN
# =====================
trainer.train()
trainer.evaluate()
