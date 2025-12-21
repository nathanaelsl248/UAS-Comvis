"""
Final Evaluation with Inference Results
Evaluates all 2000 validation images with SR outputs
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
import lpips
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd

class FinalEvaluator:
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        try:
            self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
            print("✓ LPIPS model loaded")
        except Exception as e:
            print(f"⚠ Warning: Could not load LPIPS: {e}")
            self.lpips_model = None
        
    def calculate_psnr(self, img1, img2):
        try:
            return psnr(img1, img2, data_range=255)
        except:
            return 0.0
    
    def calculate_ssim(self, img1, img2):
        try:
            return ssim(img1, img2, channel_axis=2, data_range=255)
        except:
            return 0.0
    
    def calculate_lpips(self, img1, img2):
        if self.lpips_model is None:
            return 0.0
        
        try:
            img1_t = torch.from_numpy(img1).permute(2, 0, 1).float() / 127.5 - 1
            img2_t = torch.from_numpy(img2).permute(2, 0, 1).float() / 127.5 - 1
            
            img1_t = img1_t.unsqueeze(0).to(self.device)
            img2_t = img2_t.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                lpips_score = self.lpips_model(img1_t, img2_t)
            
            return lpips_score.item()
        except:
            return 0.0
    
    def bicubic_upsample(self, lr_img, scale=4):
        h, w = lr_img.shape[:2]
        return cv2.resize(lr_img, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
    
    def evaluate(self, val_hr_dir, val_lr_dir, sr_dir, output_dir='final_evaluation_results'):
        os.makedirs(output_dir, exist_ok=True)
        
        # Get file lists
        hr_files = sorted(list(Path(val_hr_dir).glob('*.png')) + list(Path(val_hr_dir).glob('*.jpg')))
        lr_files = sorted(list(Path(val_lr_dir).glob('*.png')) + list(Path(val_lr_dir).glob('*.jpg')))
        sr_files = sorted(list(Path(sr_dir).glob('*.png')) + list(Path(sr_dir).glob('*.jpg')))
        
        print(f"\n✓ Found {len(hr_files)} HR images")
        print(f"✓ Found {len(lr_files)} LR images")
        print(f"✓ Found {len(sr_files)} SR images")
        
        results = {
            'filename': [],
            'bicubic_psnr': [], 'bicubic_ssim': [], 'bicubic_lpips': [],
            'realesrgan_psnr': [], 'realesrgan_ssim': [], 'realesrgan_lpips': []
        }
        
        print("\nEvaluating all images...")
        matched = 0
        
        for idx, hr_file in enumerate(tqdm(hr_files)):
            # Read HR
            hr = cv2.imread(str(hr_file))
            if hr is None:
                continue
            hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
            h, w = hr.shape[:2]
            
            # Read LR
            if idx < len(lr_files):
                lr = cv2.imread(str(lr_files[idx]))
                if lr is None:
                    continue
                lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
            else:
                continue
            
            results['filename'].append(hr_file.name)
            
            # Bicubic baseline
            bicubic = self.bicubic_upsample(lr)
            bicubic = bicubic[:h, :w]
            
            results['bicubic_psnr'].append(self.calculate_psnr(hr, bicubic))
            results['bicubic_ssim'].append(self.calculate_ssim(hr, bicubic))
            results['bicubic_lpips'].append(self.calculate_lpips(hr, bicubic))
            
            # Find matching SR file
            sr_file = None
            # Try exact match first
            for sf in sr_files:
                if hr_file.stem in sf.stem or sf.stem.replace('_sr', '') == hr_file.stem:
                    sr_file = sf
                    break
            
            # If not found, try index-based match
            if sr_file is None and idx < len(sr_files):
                sr_file = sr_files[idx]
            
            if sr_file and sr_file.exists():
                sr = cv2.imread(str(sr_file))
                if sr is not None:
                    sr = cv2.cvtColor(sr, cv2.COLOR_BGR2RGB)
                    sr = sr[:h, :w]
                    
                    results['realesrgan_psnr'].append(self.calculate_psnr(hr, sr))
                    results['realesrgan_ssim'].append(self.calculate_ssim(hr, sr))
                    results['realesrgan_lpips'].append(self.calculate_lpips(hr, sr))
                    matched += 1
                else:
                    results['realesrgan_psnr'].append(np.nan)
                    results['realesrgan_ssim'].append(np.nan)
                    results['realesrgan_lpips'].append(np.nan)
            else:
                results['realesrgan_psnr'].append(np.nan)
                results['realesrgan_ssim'].append(np.nan)
                results['realesrgan_lpips'].append(np.nan)
        
        print(f"\n✓ Successfully matched {matched}/{len(hr_files)} SR images")
        
        # Save detailed results
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(output_dir, 'detailed_results.csv'), index=False)
        
        # Calculate statistics
        stats = self.calculate_statistics(df)
        stats.to_csv(os.path.join(output_dir, 'statistics.csv'), index=False)
        
        # Create visualizations
        self.create_visualizations(df, output_dir)
        
        # Print results
        self.print_results(df)
        
        return df, stats
    
    def calculate_statistics(self, df):
        stats_data = []
        
        # Bicubic
        stats_data.append({
            'Method': 'BICUBIC',
            'PSNR (↑)': f"{df['bicubic_psnr'].mean():.4f} ± {df['bicubic_psnr'].std():.4f}",
            'SSIM (↑)': f"{df['bicubic_ssim'].mean():.4f} ± {df['bicubic_ssim'].std():.4f}",
            'LPIPS (↓)': f"{df['bicubic_lpips'].mean():.4f} ± {df['bicubic_lpips'].std():.4f}"
        })
        
        # Real-ESRGAN
        df_sr = df.dropna(subset=['realesrgan_psnr'])
        if len(df_sr) > 0:
            stats_data.append({
                'Method': 'REAL-ESRGAN',
                'PSNR (↑)': f"{df_sr['realesrgan_psnr'].mean():.4f} ± {df_sr['realesrgan_psnr'].std():.4f}",
                'SSIM (↑)': f"{df_sr['realesrgan_ssim'].mean():.4f} ± {df_sr['realesrgan_ssim'].std():.4f}",
                'LPIPS (↓)': f"{df_sr['realesrgan_lpips'].mean():.4f} ± {df_sr['realesrgan_lpips'].std():.4f}"
            })
        
        return pd.DataFrame(stats_data)
    
    def create_visualizations(self, df, output_dir):
        df_clean = df.dropna(subset=['realesrgan_psnr'])
        
        if len(df_clean) == 0:
            print("⚠ No SR data for visualization")
            return
        
        metrics = ['psnr', 'ssim', 'lpips']
        metric_names = ['PSNR (dB)', 'SSIM', 'LPIPS']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx]
            
            data_to_plot = [
                df[f'bicubic_{metric}'].values,
                df_clean[f'realesrgan_{metric}'].values
            ]
            labels = ['Bicubic', 'Real-ESRGAN (Ours)']
            
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            
            colors = ['lightblue', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            ax.set_ylabel(name, fontsize=12)
            ax.set_title(f'{name} Comparison', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, 'metrics_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Visualization saved: {save_path}")
    
    def print_results(self, df):
        df_clean = df.dropna(subset=['realesrgan_psnr'])
        
        print("\n" + "="*70)
        print("FINAL EVALUATION RESULTS")
        print("="*70)
        
        print("\nBICUBIC BASELINE:")
        print(f"  PSNR: {df['bicubic_psnr'].mean():.4f} ± {df['bicubic_psnr'].std():.4f} dB")
        print(f"  SSIM: {df['bicubic_ssim'].mean():.4f} ± {df['bicubic_ssim'].std():.4f}")
        print(f"  LPIPS: {df['bicubic_lpips'].mean():.4f} ± {df['bicubic_lpips'].std():.4f}")
        
        if len(df_clean) > 0:
            print("\nREAL-ESRGAN (OURS):")
            print(f"  PSNR: {df_clean['realesrgan_psnr'].mean():.4f} ± {df_clean['realesrgan_psnr'].std():.4f} dB")
            print(f"  SSIM: {df_clean['realesrgan_ssim'].mean():.4f} ± {df_clean['realesrgan_ssim'].std():.4f}")
            print(f"  LPIPS: {df_clean['realesrgan_lpips'].mean():.4f} ± {df_clean['realesrgan_lpips'].std():.4f}")
            
            # Calculate improvement
            psnr_imp = df_clean['realesrgan_psnr'].mean() - df['bicubic_psnr'].mean()
            ssim_imp = df_clean['realesrgan_ssim'].mean() - df['bicubic_ssim'].mean()
            lpips_imp = df['bicubic_lpips'].mean() - df_clean['realesrgan_lpips'].mean()
            
            print("\nIMPROVEMENT vs BICUBIC:")
            print(f"  PSNR: +{psnr_imp:.4f} dB ({psnr_imp/df['bicubic_psnr'].mean()*100:+.2f}%)")
            print(f"  SSIM: +{ssim_imp:.4f} ({ssim_imp/df['bicubic_ssim'].mean()*100:+.2f}%)")
            print(f"  LPIPS: -{lpips_imp:.4f} ({lpips_imp/df['bicubic_lpips'].mean()*100:+.2f}% better)")
        
        print("="*70)

if __name__ == "__main__":
    print("="*70)
    print("FINAL COMPREHENSIVE EVALUATION")
    print("="*70)
    
    # Paths
    VAL_HR_DIR = r"..\val\HR"
    VAL_LR_DIR = r"..\val\LR"
    SR_DIR = "inference_results_edsr"
    OUTPUT_DIR = "final_evaluation_results_edsr"
    SR_OUTPUT_DIR = r"C:\Python\Python311\Lib\site-packages\experiments\finetune_EDSR_x4\visualization"
    
    # Check paths
    if not os.path.exists(SR_DIR):
        print(f" SR directory not found: {SR_DIR}")
        exit(1)
    
    # Run evaluation
    evaluator = FinalEvaluator(device='cuda' if torch.cuda.is_available() else 'cpu')
    df, stats = evaluator.evaluate(VAL_HR_DIR, VAL_LR_DIR, SR_DIR, OUTPUT_DIR)
    
    print(f"\n✓ Evaluation complete!")
    print(f"✓ Results saved to: {OUTPUT_DIR}/")
    print(f"✓ Detailed CSV: {OUTPUT_DIR}/detailed_results.csv")
    print(f"✓ Statistics: {OUTPUT_DIR}/statistics.csv")
    print(f"✓ Visualization: {OUTPUT_DIR}/metrics_comparison.png")
