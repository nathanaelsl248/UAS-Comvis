import os

DATASET_ROOT = r"c:\Users\beast\Documents\Kuliah\Computer Vision\UASComVis\dataset\LSDIR"
HR_PATH = os.path.join(DATASET_ROOT, "HR")          
LR_PATH = os.path.join(DATASET_ROOT, "LR_X4")       
JSON_PATH = os.path.join(DATASET_ROOT, "train_X4.json")  

BATCH_SIZE = 4               
LEARNING_RATE = 1e-4         
NUM_EPOCHS = 50               
SCALE_FACTOR = 4           

PATCH_SIZE = 48             
HR_PATCH_SIZE = PATCH_SIZE * SCALE_FACTOR  

TRAIN_SPLIT = 0.98  
VAL_SPLIT = 0.02    

INPUT_CHANNELS = 3      
LATENT_DIM = 128           
BASE_FILTERS = 64         

CHECKPOINT_DIR = "results/checkpoints"
LOG_DIR = "results/logs"
OUTPUT_DIR = "results/output"
SAVE_FREQ = 5   

DEVICE = "cuda" 

USE_AUGMENTATION = True
HORIZONTAL_FLIP_PROB = 0.5
VERTICAL_FLIP_PROB = 0.5
ROTATION_PROB = 0.3

L1_WEIGHT = 1.0          
PERCEPTUAL_WEIGHT = 0    
SSIM_WEIGHT = 0.1     

MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

SAVE_COMPARISON = True    
CALCULATE_METRICS = True 

def create_dirs():
    """Buat direktori yang diperlukan jika belum ada"""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

if __name__ == "__main__":
    print("=== Konfigurasi Proyek ===")
    print(f"Dataset Root: {DATASET_ROOT}")
    print(f"HR Path: {HR_PATH}")
    print(f"LR Path: {LR_PATH}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Scale Factor: {SCALE_FACTOR}")
    create_dirs()
    print("Direktori berhasil dibuat!")
