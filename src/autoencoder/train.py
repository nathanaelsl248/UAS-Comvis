import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import logging
import datetime
import numpy as np
from typing import Optional
import torchvision.transforms.functional as TF
import torchvision.models as models
from torchvision.models import VGG19_Weights
from torch.amp import autocast, GradScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config
from src.dataset import create_dataloaders
from src.autoencoder.model import create_model, count_parameters


class VGGFeatureExtractor(nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.features = nn.Sequential(*list(vgg.children())[:17]).to(device)
        for p in self.features.parameters():
            p.requires_grad = False
        self.register_buffer('mean', torch.as_tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1))
        self.register_buffer('std', torch.as_tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.features(x)


def ssim_loss(x, y, window_size=11, C1=0.01 ** 2, C2=0.03 ** 2):
    padding = window_size // 2
    mu_x = F.avg_pool2d(x, window_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(y, window_size, stride=1, padding=padding)

    mu_x2 = mu_x.mul(mu_x)
    mu_y2 = mu_y.mul(mu_y)
    mu_xy = mu_x.mul(mu_y)

    sigma_x2 = F.avg_pool2d(x.mul(x), window_size, 1, padding).sub_(mu_x2)
    sigma_y2 = F.avg_pool2d(y.mul(y), window_size, 1, padding).sub_(mu_y2)
    sigma_xy = F.avg_pool2d(x.mul(y), window_size, 1, padding).sub_(mu_xy)

    ssim_n = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    ssim_d = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim_map = ssim_n / ssim_d.clamp_(min=1e-7)
    return 1 - ssim_map.mean()


class CombinedLoss(nn.Module):
    def __init__(self, device, l1_weight=1.0, perceptual_weight=0.05, ssim_weight=0.1):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.l1_w = float(l1_weight)
        self.perc_w = perceptual_weight
        self.ssim_w = ssim_weight
        self.use_perc = perceptual_weight > 0
        self.use_ssim = ssim_weight > 0
        self.vgg = VGGFeatureExtractor(device) if self.use_perc else None

    def forward(self, pred, target):
        loss = self.l1_w * self.l1(pred, target)

        if self.use_perc:
            feat_p = self.vgg(pred)
            feat_t = self.vgg(target)
            perc = F.l1_loss(feat_p, feat_t)
            loss = loss + self.perc_w * perc

        if self.use_ssim:
            ssim = ssim_loss(pred, target)
            loss = loss + self.ssim_w * ssim

        return loss

log = logging.getLogger("train")


def train_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp):
    model.train()
    total_loss = 0.0
    non_blocking = device.type == 'cuda'
    
    pbar = tqdm(train_loader, desc="  Train", leave=False)
    for lr_imgs, hr_imgs in pbar:
        lr_imgs = lr_imgs.to(device, non_blocking=non_blocking)
        hr_imgs = hr_imgs.to(device, non_blocking=non_blocking)
        
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type='cuda', enabled=use_amp):
            sr_imgs = model(lr_imgs)
            loss = criterion(sr_imgs, hr_imgs)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        loss_val = loss.detach().item()
        total_loss += loss_val
        pbar.set_postfix(loss=f'{loss_val:.6f}')
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device, use_amp):
    model.eval()
    total_loss = 0.0
    non_blocking = device.type == 'cuda'
    
    pbar = tqdm(val_loader, desc="  Val", leave=False)
    with torch.no_grad():
        for lr_imgs, hr_imgs in pbar:
            lr_imgs = lr_imgs.to(device, non_blocking=non_blocking)
            hr_imgs = hr_imgs.to(device, non_blocking=non_blocking)
            with autocast(device_type='cuda', enabled=use_amp):
                sr_imgs = model(lr_imgs)
                loss = criterion(sr_imgs, hr_imgs)
            loss_val = loss.item()
            total_loss += loss_val
            pbar.set_postfix(loss=f'{loss_val:.6f}')
    
    return total_loss / len(val_loader)


def visualize_results(model, val_loader, device, epoch):
    ds = val_loader.dataset
    item = ds.data[0]

    lr_basename = os.path.basename(item['path_lq']) if isinstance(item, dict) and 'path_lq' in item else None
    if lr_basename is None:
        raise RuntimeError("Metadata JSON tidak memiliki field 'path_lq'.")
    lr_path = os.path.join(ds.dataset_root, 'LR_X4', lr_basename)
    lr_img = Image.open(lr_path).convert('RGB')
    lr_np = np.asarray(lr_img, dtype=np.float32) / 255.0

    lr_tensor = TF.to_tensor(lr_img).unsqueeze(0).to(device)

    with torch.no_grad():
        sr_tensor = model(lr_tensor)

    sr_np = sr_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    sr_np = np.clip(sr_np, 0, 1)
    lr_np = np.clip(lr_np, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(lr_np)
    axes[0].set_title('Sebelum (LR Input)')
    axes[0].axis('off')

    axes[1].imshow(sr_np)
    axes[1].set_title(f'Sesudah (SR Output Epoch {epoch})')
    axes[1].axis('off')

    plt.tight_layout()

    save_path = os.path.join(config.OUTPUT_DIR, f'sample_epoch_{epoch}.png')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show(block=False)
    plt.pause(1)
    plt.close()



def train(finetune_from: Optional[str] = None,
          resume_from: Optional[str] = None,
          lr_override: Optional[float] = None,
          freeze_encoder_epochs: int = 0):
    config.create_dirs()

    log_path = os.path.join(
        config.LOG_DIR,
        f"training_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"
    )

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )

    file_handler = None

    log.info("\n" + "="*60)
    log.info("AUTOENCODER SUPER-RESOLUTION - TRAINING")
    log.info("="*60 + "\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    log.info(f"Device: {device}")
    if device.type == 'cuda':
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"CUDA Version: {torch.version.cuda}")
        log.info(f"PyTorch: {torch.__version__}")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        log.info(f"cuDNN enabled: True, benchmark: True")
        gpu_props = torch.cuda.get_device_properties(0)
        log.info(f"GPU Memory: {gpu_props.total_memory / 1024**3:.1f} GB")
        log.info(f"Batch Size: {config.BATCH_SIZE}, Patch: {config.PATCH_SIZE}\n")
    else:
        log.warning(f"WARNING: Using CPU (slow). GPU not detected.\n")

    if not os.path.isdir(config.DATASET_ROOT):
        log.error(f"Dataset root not found: {config.DATASET_ROOT}")
        log.error("Please set DATASET_ROOT in config.py to your LSDIR path.")
        return
    if not os.path.isfile(config.JSON_PATH):
        log.error(f"JSON metadata not found: {config.JSON_PATH}")
        log.error("Expected a JSON listing pairs for LR_X4 and HR images (e.g., train_X4.json).")
        return
    hr_dir = os.path.join(config.DATASET_ROOT, 'HR')
    lr_dir = os.path.join(config.DATASET_ROOT, 'LR_X4')
    if not os.path.isdir(hr_dir) or not os.path.isdir(lr_dir):
        log.error("Expected folders 'HR' and 'LR_X4' under DATASET_ROOT.")
        log.error(f"HR dir exists: {os.path.isdir(hr_dir)} | LR_X4 dir exists: {os.path.isdir(lr_dir)}")
        return

    log.info("Loading dataset...")
    train_loader, val_loader = create_dataloaders(config)

    log.info("Creating model...")
    model = create_model(config).to(device)
    total_params = count_parameters(model)
    log.info(f"  Parameters: {total_params:,} ({total_params*4/1e6:.1f}MB)\n")

    criterion = CombinedLoss(
        device=device,
        l1_weight=config.L1_WEIGHT,
        perceptual_weight=config.PERCEPTUAL_WEIGHT,
        ssim_weight=config.SSIM_WEIGHT,
    )
    effective_lr = lr_override if lr_override is not None else config.LEARNING_RATE
    optimizer = optim.Adam(model.parameters(), lr=effective_lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    scaler = GradScaler(enabled=use_amp)

    start_epoch = 1
    if resume_from:
        log.info(f"Resuming from checkpoint: {resume_from}")
        from src.utils import load_checkpoint
        last_epoch, _ = load_checkpoint(model, resume_from, optimizer=optimizer, device=device)
        for _ in range(max(0, last_epoch)):
            scheduler.step()
        start_epoch = last_epoch + 1
    elif finetune_from:
        log.info(f"Fine-tuning from weights: {finetune_from}")
        from src.utils import load_checkpoint
        _ , _ = load_checkpoint(model, finetune_from, optimizer=None, device=device)
        for g in optimizer.param_groups:
            g['lr'] = effective_lr

    best_loss = float('inf')
    best_epoch = 0
    log.info(f"Training: {config.NUM_EPOCHS} epochs, batch_size={config.BATCH_SIZE}\n")
    
    for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
        if freeze_encoder_epochs > 0 and epoch <= freeze_encoder_epochs:
            for p in model.encoder.parameters():
                p.requires_grad = False
        else:
            for p in model.encoder.parameters():
                p.requires_grad = True
        log.info(f"Epoch [{epoch}/{config.NUM_EPOCHS}]")

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp)
        log.info(f"  Train Loss: {train_loss:.6f}")

        if epoch % 5 == 0 or epoch == 1:
            val_loss = validate(model, val_loader, criterion, device, use_amp)
            log.info(f"  Val Loss:   {val_loss:.6f}")

            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
                best_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
                torch.save(model.state_dict(), best_path)
                log.info(f"Best model saved\n")
        else:
            log.info(f"  (Skipped validation)\n")
        
        scheduler.step()

        if epoch == config.NUM_EPOCHS or epoch % 15 == 0:
            visualize_results(model, val_loader, device, epoch)
            log.info(f"  ✓ Visualization saved\n")

    log.info("="*60)
    log.info("TRAINING COMPLETE")
    log.info("="*60)
    log.info(f"\nBest Val Loss: {best_loss:.6f} (Epoch {best_epoch})")
    log.info(f"Model: {os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')}")

    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("TRAINING SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Epochs: {config.NUM_EPOCHS}\n")
        f.write(f"  Batch Size: {config.BATCH_SIZE}\n")
        f.write(f"  Patch Size: {config.PATCH_SIZE} (HR: {config.HR_PATCH_SIZE})\n")
        f.write(f"  Model Parameters: {total_params:,}\n\n")
        f.write(f"Results:\n")
        f.write(f"  Best Validation Loss: {best_loss:.6f}\n")
        f.write(f"  Best Epoch: {best_epoch}\n")
        f.write(f"  Model Path: {os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')}\n\n")
        f.write("="*60 + "\n")
    
    log.info(f"\nSummary saved to: {log_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Autoencoder Super-Resolution")
    parser.add_argument("--finetune-from", type=str, default=None, help="Path weights/ckpt untuk fine-tuning (reset optimizer)")
    parser.add_argument("--resume-from", type=str, default=None, help="Path checkpoint untuk resume (lanjut optimizer)")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--freeze-encoder-epochs", type=int, default=0, help="Bekukan encoder N epoch awal")
    args = parser.parse_args()

    train(
        finetune_from=args.finetune_from,
        resume_from=args.resume_from,
        lr_override=args.lr,
        freeze_encoder_epochs=args.freeze_encoder_epochs,
    )
