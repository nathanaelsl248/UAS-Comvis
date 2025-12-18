import os
import sys
import torch
import argparse
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config
from src.autoencoder.model import create_model
from src.utils import (
    calculate_metrics,
    load_checkpoint,
    visualize_comparison,
    save_image,
    tensor_to_image,
    AverageMeter
)

#single image inference function
def load_model(model_path, device):
    print(f"Loading model from: {model_path}")
    model = create_model(config).to(device)

    if os.path.exists(model_path):
        load_checkpoint(model, model_path, device=device)

    model.eval()

    return model


def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    original_size = img.size
    img_tensor = TF.to_tensor(img).unsqueeze(0)

    return img_tensor, original_size


def super_resolve_image(model, lr_image_path, device, save_path=None):
    lr_tensor, original_size = preprocess_image(lr_image_path)
    lr_tensor = lr_tensor.to(device)
    
    print(f"Input image size: {original_size}")
    print(f"Input tensor shape: {lr_tensor.shape}")

    with torch.no_grad():
        sr_tensor = model(lr_tensor)
    
    print(f"Output tensor shape: {sr_tensor.shape}")
    if save_path:
        save_image(sr_tensor[0], save_path)
        print(f"Saved to: {save_path}")
    
    return sr_tensor


#batch inference function
def batch_inference(model, input_dir, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)
    image_extensions = ('.png')
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(image_extensions)]
    
    for img_file in tqdm(image_files, desc="Processing images"):
        input_path = os.path.join(input_dir, img_file)
        output_path = os.path.join(output_dir, f"sr_{img_file}")
        
        try:
            super_resolve_image(model, input_path, device, output_path)
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
    
    print(f"\nAll images saved to: {output_dir}")


#evaluation function
def evaluate_model(model, test_json_path, dataset_root, device, save_dir=None):
    print("Evaluating model...")
    with open(test_json_path, 'r') as f:
        test_data = json.load(f)
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    def resolve_dataset_path(root, subdir, p):
        if os.path.isabs(p) and os.path.exists(p):
            return p

        base = os.path.basename(p)
        cand = os.path.join(root, subdir, base)
        if os.path.exists(cand):
            return cand

        cand2 = os.path.join(root, p)
        if os.path.exists(cand2):
            return cand2

        cand3 = os.path.join(root, p.replace('X4/', 'LR_X4/').replace('LR/', 'LR_X4/'))
        if os.path.exists(cand3):
            return cand3

        return cand

    for idx, item in enumerate(tqdm(test_data, desc="Evaluating")):
        hr_path = resolve_dataset_path(dataset_root, 'HR', item['path_gt'])
        lr_path = resolve_dataset_path(dataset_root, 'LR_X4', item['path_lq'])
        
        try:
            lr_tensor, _ = preprocess_image(lr_path)
            hr_tensor, _ = preprocess_image(hr_path)
            
            lr_tensor = lr_tensor.to(device)
            hr_tensor = hr_tensor.to(device)

            with torch.no_grad():
                sr_tensor = model(lr_tensor)

            metrics = calculate_metrics(sr_tensor[0], hr_tensor[0])
            psnr_meter.update(metrics['psnr'])
            ssim_meter.update(metrics['ssim'])

            if save_dir and idx < 10:
                save_path = os.path.join(save_dir, f'comparison_{idx}.png')
                visualize_comparison(lr_tensor[0], sr_tensor[0], hr_tensor[0], save_path)
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue

    results = {
        'avg_psnr': psnr_meter.avg,
        'avg_ssim': ssim_meter.avg,
        'num_samples': psnr_meter.count
    }
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Number of samples: {results['num_samples']}")
    print(f"Average PSNR: {results['avg_psnr']:.2f} dB")
    print(f"Average SSIM: {results['avg_ssim']:.4f}")
    print("="*60)
    
    return results

#perbandingan dengan ground truth
def compare_with_gt(model, lr_image_path, hr_image_path, device, save_path=None):
    print(f"Comparing with ground truth...")

    lr_tensor, _ = preprocess_image(lr_image_path)
    hr_tensor, _ = preprocess_image(hr_image_path)
    
    lr_tensor = lr_tensor.to(device)
    hr_tensor = hr_tensor.to(device)

    with torch.no_grad():
        sr_tensor = model(lr_tensor)

    metrics = calculate_metrics(sr_tensor[0], hr_tensor[0])
    
    print(f"PSNR: {metrics['psnr']:.2f} dB")
    print(f"SSIM: {metrics['ssim']:.4f}")

    if save_path:
        visualize_comparison(lr_tensor[0], sr_tensor[0], hr_tensor[0], save_path)
    
    return metrics

def demo_mode(model, device):
    print("\n" + "="*60)
    print("INTERACTIVE DEMO MODE")
    print("="*60)
    print("Enter 'quit' to exit")
    
    while True:
        lr_path = input("\nEnter path to LR image: ").strip()
        
        if lr_path.lower() == 'quit':
            break
        
        if not os.path.exists(lr_path):
            print(f"File not found: {lr_path}")
            continue

        output_path = input("Enter output path (or press Enter for default): ").strip()
        if not output_path:
            basename = os.path.basename(lr_path)
            output_path = os.path.join(config.OUTPUT_DIR, f"sr_{basename}")
        
        try:
            super_resolve_image(model, lr_path, device, output_path)

            compare = input("Compare with ground truth? (y/n): ").strip().lower()
            if compare == 'y':
                hr_path = input("Enter path to HR image: ").strip()
                if os.path.exists(hr_path):
                    compare_path = output_path.replace('.png', '_comparison.png')
                    compare_with_gt(model, lr_path, hr_path, device, compare_path)
        
        except Exception as e:
            print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Inference for Super-Resolution Autoencoder')

    parser.add_argument('--mode', type=str, default='single', 
                       choices=['single', 'batch', 'evaluate', 'demo'],
                       help='Inference mode')

    parser.add_argument('--model', type=str, default=config.MODEL_PATH,
                       help='Path to trained model')

    parser.add_argument('--input', type=str, default=None,
                       help='Input image path (single mode) or directory (batch mode)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path or directory')

    parser.add_argument('--test-json', type=str, default=None,
                       help='Path to test JSON file (evaluate mode)')
    parser.add_argument('--hr-image', type=str, default=None,
                       help='Ground truth HR image for comparison')
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and config.DEVICE == 'cuda' else 'cpu')
    print(f"Using device: {device}")

    model = load_model(args.model, device)

    if args.mode == 'single':
        if not args.input:
            print("Error: --input required for single mode")
            return
        
        output = args.output or os.path.join(config.OUTPUT_DIR, 
                                             f"sr_{os.path.basename(args.input)}")
        
        super_resolve_image(model, args.input, device, output)

        if args.hr_image:
            compare_path = output.replace('.png', '_comparison.png')
            compare_with_gt(model, args.input, args.hr_image, device, compare_path)
    
    elif args.mode == 'batch':
        if not args.input or not args.output:
            print("Error: --input and --output required for batch mode")
            return
        
        batch_inference(model, args.input, args.output, device)
    
    elif args.mode == 'evaluate':
        if not args.test_json:
            print("Error: --test-json required for evaluate mode")
            return
        
        save_dir = args.output or os.path.join(config.OUTPUT_DIR, 'evaluation')
        evaluate_model(model, args.test_json, config.DATASET_ROOT, device, save_dir)
    
    elif args.mode == 'demo':
        demo_mode(model, device)
    
    print("\nInference completed!")

if __name__ == "__main__":
    main()
