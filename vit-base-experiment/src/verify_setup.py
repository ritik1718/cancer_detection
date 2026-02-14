import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Fix: Import local config directly to avoid shadowing by root src
try:
    from config import Config
    from model import get_model
except ImportError:
    # Fallback if running from root
    from src.config import Config
    from src.model import get_model
    
from src.dataset import OralCancerDataset, get_transforms
from src.dataset import OralCancerDataset, get_transforms
from torch.utils.data import DataLoader
import torch

def verify():
    print("="*50)
    print("Verifying ViT Experiment Setup (384px)")
    print("="*50)
    
    # 1. Check Config
    print(f"Model: {Config.MODEL_NAME}")
    print(f"Img Size: {Config.IMG_SIZE}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Data Dir: {Config.DATA_DIR}")
    
    if Config.IMG_SIZE != 384:
        print("❌ Error: IMG_SIZE should be 384!")
        return
        
    if not os.path.exists(Config.DATA_DIR):
        print(f"❌ Error: Data directory not found at {Config.DATA_DIR}")
        return
    print("✅ Configuration looks correct.")

    # 2. Check Model Loading
    print("\nLoading Model...")
    try:
        model = get_model(num_classes=Config.NUM_CLASSES, model_name=Config.MODEL_NAME, pretrained=False)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 3. Check Dataset Access
    print("\nChecking Dataset Access...")
    try:
        dataset = OralCancerDataset(Config.TRAIN_DIR, transform=get_transforms(train=True, img_size=Config.IMG_SIZE))
        print(f"✅ Dataset initialized. Found {len(dataset)} images.")
        
        if len(dataset) > 0:
            img, label = dataset[0]
            print(f"✅ Sample image shape: {img.shape}")
            if img.shape[1] != 384 or img.shape[2] != 384:
                 print(f"❌ Error: Image shape mismatch! Expected 384x384, got {img.shape[1:]}")
    except Exception as e:
        print(f"❌ Error accessing dataset: {e}")
        return

    print("\n✅ Verification Complete!")

if __name__ == "__main__":
    verify()
