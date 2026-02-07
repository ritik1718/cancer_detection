import sys
import os
import torch

# Add the current directory to sys.path so we can import from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.model import get_model
from src.dataset import OralCancerDataset, get_transforms
from torch.utils.data import DataLoader

def verify():
    print("Verifying Swin Transformer Setup...")
    
    # 1. Check Model
    print("\n1. Initializing Swin Transformer model...")
    try:
        model = get_model(num_classes=Config.NUM_CLASSES, pretrained=False, model_name=Config.MODEL_NAME)
        print("✅ Model initialized successfully!")
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        return

    # 2. Check Data Loading (Mock or Real)
    print("\n2. Checking Data Loading...")
    try:
        # Create a dummy dataset check
        if os.path.exists(Config.TRAIN_DIR):
            dataset = OralCancerDataset(Config.TRAIN_DIR, transform=get_transforms(train=True, img_size=Config.IMG_SIZE))
            if len(dataset) > 0:
                print(f"✅ Dataset found with {len(dataset)} images.")
                loader = DataLoader(dataset, batch_size=2, shuffle=True)
                images, labels = next(iter(loader))
                print(f"✅ Batch loaded: Images {images.shape}, Labels {labels.shape}")
                
                # 3. Forward Pass
                print("\n3. Running Forward Pass...")
                if torch.cuda.is_available():
                    model = model.cuda()
                    images = images.cuda()
                
                outputs = model(images)
                print(f"✅ Forward pass successful. Output shape: {outputs.shape}")
            else:
                print("⚠️ Dataset directory exists but is empty.")
        else:
            print(f"⚠️ Dataset directory {Config.TRAIN_DIR} not found. Skipping data check.")
            
    except Exception as e:
        print(f"❌ Data loading/Forward pass failed: {e}")
        return

    print("\n✅ SETUP VERIFIED SUCCESSFULLY!")

if __name__ == "__main__":
    verify()
