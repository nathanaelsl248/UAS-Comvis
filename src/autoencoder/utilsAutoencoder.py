import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import torchvision.transforms.functional as TF


def calculate_psnr(img1, img2, max_val=1.0):
    if torch.is_tensor(img1):
        img1 = img1.detach().cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.detach().cpu().numpy()

    if img1.ndim == 3 and img1.shape[0] == 3:
        img1 = np.transpose(img1, (1, 2, 0))
    if img2.ndim == 3 and img2.shape[0] == 3:
        img2 = np.transpose(img2, (1, 2, 0))

    return psnr(img1, img2, data_range=max_val)


def calculate_ssim(img1, img2, max_val=1.0):
    if torch.is_tensor(img1):
        img1 = img1.detach().cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.detach().cpu().numpy()

    if img1.ndim == 3 and img1.shape[0] == 3:
        img1 = np.transpose(img1, (1, 2, 0))
    if img2.ndim == 3 and img2.shape[0] == 3:
        img2 = np.transpose(img2, (1, 2, 0))

    return ssim(img1, img2, data_range=max_val, channel_axis=-1)


def calculate_metrics(sr_img, hr_img):
    psnr_val = calculate_psnr(sr_img, hr_img)
    ssim_val = calculate_ssim(sr_img, hr_img)
    
    return {
        'psnr': psnr_val,
        'ssim': ssim_val
    }

def tensor_to_image(tensor):
    if tensor.ndim == 4:
        tensor = tensor[0]

    img = tensor.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0, 1) * 255
    img = img.astype(np.uint8)
    
    return img


def visualize_comparison(lr_img, sr_img, hr_img, save_path=None):
    lr_np = tensor_to_image(lr_img)
    sr_np = tensor_to_image(sr_img)
    hr_np = tensor_to_image(hr_img)

    metrics = calculate_metrics(sr_img, hr_img)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(lr_np)
    axes[0].set_title('Low Resolution (Input)', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(sr_np)
    axes[1].set_title(f'Super-Resolved\nPSNR: {metrics["psnr"]:.2f} dB\nSSIM: {metrics["ssim"]:.4f}', 
                     fontsize=12)
    axes[1].axis('off')
    
    axes[2].imshow(hr_np)
    axes[2].set_title('High Resolution (Ground Truth)', fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison saved to: {save_path}")
    
    plt.show()
    plt.close()


def plot_training_curves(train_losses, val_losses, save_path=None):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    
    plt.show()
    plt.close()


def save_checkpoint(model, optimizer, epoch, loss, save_path, is_best=False):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: {save_path}")

    if is_best:
        best_path = os.path.join(os.path.dirname(save_path), 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"Best model saved: {best_path}")


def load_checkpoint(model, checkpoint_path, optimizer=None, device='cpu'):
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return 0, float('inf')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', float('inf'))
        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"Resuming from epoch {epoch} with loss {loss:.4f}")
        return epoch, loss
    else:
        model.load_state_dict(checkpoint)
        print(f"Model weights loaded (state_dict): {checkpoint_path}")
        return 0, float('inf')

def save_image(tensor, save_path):
    img = tensor_to_image(tensor)
    img_pil = Image.fromarray(img)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save
    img_pil.save(save_path)
    print(f"Image saved: {save_path}")


def load_image(image_path, to_tensor=True):
    img = Image.open(image_path).convert('RGB')
    
    if to_tensor:
        return TF.to_tensor(img)
    
    return img

class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def print_metrics(epoch, train_loss, val_loss, psnr, ssim):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch:3d}")
    print(f"{'-'*60}")
    print(f"Train Loss: {train_loss:.6f}")
    print(f"Val Loss:   {val_loss:.6f}")
    print(f"PSNR:       {psnr:.2f} dB")
    print(f"SSIM:       {ssim:.4f}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    print("Testing utility functions...")

    img1 = torch.rand(3, 256, 256)
    img2 = img1 + torch.rand(3, 256, 256) * 0.1
    
    psnr_val = calculate_psnr(img1, img2)
    ssim_val = calculate_ssim(img1, img2)
    
    print(f"PSNR: {psnr_val:.2f} dB")
    print(f"SSIM: {ssim_val:.4f}")

    meter = AverageMeter()
    for i in range(10):
        meter.update(i)
    
    print(f"Average: {meter.avg:.2f}")
    
    print("\nUtility functions working correctly!")
