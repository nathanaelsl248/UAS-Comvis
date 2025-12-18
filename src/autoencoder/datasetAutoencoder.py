"""
Dataset Loader untuk LSDIR (Large Scale Dataset for Image Restoration)
=======================================================================
File ini berisi implementasi PyTorch Dataset untuk memuat pasangan gambar
Low Resolution (LR) dan High Resolution (HR) dari dataset LSDIR.
"""

import os
import json
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


# ===========================
# CELL 1: Dataset Class
# ===========================

class LSdirDataset(Dataset):
    """
    PyTorch Dataset untuk LSDIR
    
    Dataset ini memuat pasangan gambar LR dan HR untuk training super-resolution.
    Mendukung augmentasi data (flip, rotation) untuk meningkatkan generalisasi model.
    
    Args:
        json_path: Path ke file JSON yang berisi daftar pasangan gambar
        dataset_root: Root directory dataset
        patch_size: Ukuran patch yang akan di-crop (default: 64)
        scale: Faktor upscaling (default: 4)
        augment: Aktifkan augmentasi data (default: False)
        mode: Mode dataset ('train' atau 'val')
    """
    
    def __init__(self, json_path, dataset_root, patch_size=64, scale=4, 
                 augment=False, mode='train'):
        super().__init__()
        self.dataset_root = dataset_root
        self.patch_size = patch_size
        self.hr_patch_size = patch_size * scale
        self.scale = scale
        self.augment = augment
        self.mode = mode
        
        # Muat daftar gambar dari JSON
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        print(f"[{mode.upper()}] Loaded {len(self.data)} image pairs")
    
    def __len__(self):
        """Kembalikan jumlah sampel dalam dataset"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Ambil satu sampel dari dataset
        
        Returns:
            lr_img: Low resolution image tensor [C, H, W]
            hr_img: High resolution image tensor [C, H*scale, W*scale]
        """
        # Ambil path gambar dari metadata
        item = self.data[idx]
        
        # Path dalam JSON: "HR/train/0084000/0083571.png" 
        # File sebenarnya: HR/0083571.png (flat structure)
        # Ekstrak hanya nama file dari path JSON
        hr_filename = os.path.basename(item['path_gt'])
        lr_filename = os.path.basename(item['path_lq'])
        
        # Konstruksi path ke file sebenarnya
        hr_path = os.path.join(self.dataset_root, 'HR', hr_filename)
        lr_path = os.path.join(self.dataset_root, 'LR_X4', lr_filename)
        
        # Muat gambar
        try:
            hr_img = Image.open(hr_path).convert('RGB')
            lr_img = Image.open(lr_path).convert('RGB')
        except Exception as e:
            print(f"Error loading images: {e}")
            print(f"HR path: {hr_path}")
            print(f"LR path: {lr_path}")
            # Return gambar kosong jika error
            return torch.zeros(3, self.patch_size, self.patch_size), \
                   torch.zeros(3, self.hr_patch_size, self.hr_patch_size)
        
        # Crop patch secara random untuk training
        if self.mode == 'train':
            lr_img, hr_img = self._get_patch(lr_img, hr_img)
        else:
            # Samakan ukuran untuk validation agar batch bisa di-stack
            lr_img = lr_img.resize((self.patch_size, self.patch_size), Image.BICUBIC)
            hr_img = hr_img.resize((self.hr_patch_size, self.hr_patch_size), Image.BICUBIC)
        
        # Augmentasi data
        if self.augment:
            lr_img, hr_img = self._augment(lr_img, hr_img)
        
        # Konversi ke tensor dan normalisasi [0, 1]
        lr_tensor = TF.to_tensor(lr_img)
        hr_tensor = TF.to_tensor(hr_img)
        
        return lr_tensor, hr_tensor
    
    def _get_patch(self, lr_img, hr_img):
        """Crop patch random dari gambar untuk training - optimized"""
        lr_w, lr_h = lr_img.size
        
        if lr_w < self.patch_size or lr_h < self.patch_size:
            lr_img = lr_img.resize((self.patch_size, self.patch_size), Image.BICUBIC)
            hr_img = hr_img.resize((self.hr_patch_size, self.hr_patch_size), Image.BICUBIC)
            return lr_img, hr_img
        
        lr_x = random.randint(0, lr_w - self.patch_size)
        lr_y = random.randint(0, lr_h - self.patch_size)
        
        hr_x = lr_x * self.scale
        hr_y = lr_y * self.scale
        
        lr_patch = lr_img.crop((lr_x, lr_y, lr_x + self.patch_size, lr_y + self.patch_size))
        hr_patch = hr_img.crop((hr_x, hr_y, hr_x + self.hr_patch_size, hr_y + self.hr_patch_size))
        
        return lr_patch, hr_patch
    
    def _augment(self, lr_img, hr_img):
        """Augmentasi data: horizontal flip, vertical flip, rotation - optimized"""
        if random.random() < 0.5:
            lr_img = TF.hflip(lr_img)
            hr_img = TF.hflip(hr_img)
        
        if random.random() < 0.5:
            lr_img = TF.vflip(lr_img)
            hr_img = TF.vflip(hr_img)
        
        if random.random() < 0.3:
            angle = random.choice([90, 180, 270])
            lr_img = TF.rotate(lr_img, angle)
            hr_img = TF.rotate(hr_img, angle)
        
        return lr_img, hr_img


# ===========================
# CELL 2: Data Split Function
# ===========================

def split_dataset(json_path, train_ratio=0.9):
    """Split dataset menjadi training dan validation set - optimized"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    random.seed(42)
    random.shuffle(data)
    
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    print(f"Total samples: {len(data)} | Train: {len(train_data)} | Val: {len(val_data)}")
    
    return train_data, val_data


# ===========================
# CELL 3: DataLoader Creation
# ===========================

def create_dataloaders(config):
    """
    Buat PyTorch DataLoader untuk training dan validation
    
    Args:
        config: Configuration object/module dengan parameter dataset
        
    Returns:
        train_loader: DataLoader untuk training
        val_loader: DataLoader untuk validation
    """
    from torch.utils.data import DataLoader
    
    # Split dataset
    train_data, val_data = split_dataset(config.JSON_PATH, config.TRAIN_SPLIT)
    
    # Buat temporary JSON files untuk train dan val
    import tempfile
    train_json = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
    val_json = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
    
    json.dump(train_data, train_json)
    json.dump(val_data, val_json)
    
    train_json.close()
    val_json.close()
    
    # Buat datasets
    train_dataset = LSdirDataset(
        json_path=train_json.name,
        dataset_root=config.DATASET_ROOT,
        patch_size=config.PATCH_SIZE,
        scale=config.SCALE_FACTOR,
        augment=config.USE_AUGMENTATION,
        mode='train'
    )
    
    val_dataset = LSdirDataset(
        json_path=val_json.name,
        dataset_root=config.DATASET_ROOT,
        patch_size=config.PATCH_SIZE,
        scale=config.SCALE_FACTOR,
        augment=False,
        mode='val'
    )
    
    use_cuda = torch.cuda.is_available()
    num_workers = 6 if use_cuda else 0  # Increased untuk parallel loading
    pin_memory = True if use_cuda else False
    persistent_workers = True if use_cuda else False  # Keep workers alive untuk speed

    # Buat dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    
    # Cleanup temporary files
    os.unlink(train_json.name)
    os.unlink(val_json.name)
    
    return train_loader, val_loader


# ===========================
# TESTING
# ===========================

if __name__ == "__main__":
    # Test dataset loading
    print("Testing dataset loading...")
    
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    import config
    
    # Test dataset creation
    try:
        train_data, val_data = split_dataset(config.JSON_PATH, config.TRAIN_SPLIT)
        
        # Create temporary JSON for testing
        import tempfile
        test_json = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        json.dump(train_data[:100], test_json)  # Use only 100 samples for testing
        test_json.close()
        
        dataset = LSdirDataset(
            json_path=test_json.name,
            dataset_root=config.DATASET_ROOT,
            patch_size=config.PATCH_SIZE,
            scale=config.SCALE_FACTOR,
            augment=True,
            mode='train'
        )
        
        # Test loading one sample
        lr, hr = dataset[0]
        print(f"\nSample loaded successfully!")
        print(f"LR shape: {lr.shape}")  # Should be [3, 64, 64]
        print(f"HR shape: {hr.shape}")  # Should be [3, 256, 256]
        print(f"LR range: [{lr.min():.3f}, {lr.max():.3f}]")
        print(f"HR range: [{hr.min():.3f}, {hr.max():.3f}]")
        
        # Cleanup
        os.unlink(test_json.name)
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
