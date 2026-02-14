import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.model import get_model
except ImportError:
    # Fallback if running from a different location
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.model import get_model

def get_transforms(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

def load_model_from_checkpoint(model_name, checkpoint_path, num_classes=2, device='cuda'):
    print(f"🔄 Loading {model_name} from {checkpoint_path}...")
    
    # Initialize model architecture
    try:
        model = get_model(num_classes=num_classes, pretrained=False, model_name=model_name)
    except Exception as e:
        print(f"❌ Error creating model architecture for {model_name}: {e}")
        raise e
    
    # Load weights
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Handle possible data parallel prefix and mismatched keys
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
                
        model.load_state_dict(new_state_dict, strict=False)
        print(f"✅ Successfully loaded weights for {model_name}")
    except Exception as e:
        print(f"❌ Error loading weights for {model_name}: {e}")
        raise e

    model = model.to(device)
    model.eval()
    return model

def plot_confusion_matrix(cm, classes, output_path='confusion_matrix_ensemble.png'):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Ensemble Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"📊 Confusion matrix saved to {output_path}")

def main():
    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------
    TEST_DIR = r'c:\Projects\oral-cancer-detection\data\raw\test'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 8
    NUM_WORKERS = 0  # Avoid Windows multiprocessing bugs
    
    print("="*60)
    print("🔬 ENSEMBLE EVALUATION: Local + Global Features (DenseNet + ViT)")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Test Directory: {TEST_DIR}")
    
    # Define models to ensemble
    models_config = [
        {
            'name': 'densenet121',
            'path': r'c:\Projects\oral-cancer-detection\densenet-experiment\models\saved_models\best_model_densenet.pth',
            'img_size': 224,
            'weight': 1.0,  
            'desc': 'DenseNet121 (Local Features)'
        },
        {
            'name': 'vit_base_384',
            'path': r'c:\Projects\oral-cancer-detection\vit-base-experiment\models\saved_models\best_model_vit_base.pth',
            'img_size': 384,
            'weight': 2.4,  
            'desc': 'ViT Base (Global Features)'
        }
    ]
    
    # ---------------------------------------------------------
    # Load Models
    # ---------------------------------------------------------
    loaded_models = []
    print("\nLoading Models...")
    for config in models_config:
        try:
            if not os.path.exists(config['path']):
                print(f"⚠️ Warning: Checkpoint not found at {config['path']}")
                continue
                
            model = load_model_from_checkpoint(config['name'], config['path'], device=DEVICE)
            loaded_models.append({
                'model': model,
                'img_size': config['img_size'],
                'weight': config['weight'],
                'name': config['name'],
                'desc': config['desc']
            })
        except Exception as e:
            print(f"❌ SKIPPING {config['name']} due to error: {e}")

    if not loaded_models:
        print("❌ No models loaded. Exiting.")
        return

    # ---------------------------------------------------------
    # Prepare Data Loaders
    # ---------------------------------------------------------
    loaders = {}
    dataset_sizes = set()
    classes = None
    
    unique_sizes = set(m['img_size'] for m in loaded_models)
    
    print("\nPreparing DataLoaders...")
    for size in unique_sizes:
        try:
            dataset = datasets.ImageFolder(TEST_DIR, transform=get_transforms(size))
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
            loaders[size] = loader
            dataset_sizes.add(len(dataset))
            if classes is None:
                classes = dataset.classes
            print(f"✅ Created dataloader for image size {size}x{size}")
        except Exception as e:
            print(f"❌ Error creating dataloader for size {size}: {e}")
            return

    if not classes:
        print("❌ Could not find classes. Check data directory.")
        return
        
    print(f"Classes found: {classes}")
    print(f"Total test images: {list(dataset_sizes)[0]}")
    
    # ---------------------------------------------------------
    # Evaluation Loop
    # ---------------------------------------------------------
    all_preds = []
    all_labels = []
    
    loader_list = [loaders[size] for size in sorted(unique_sizes)]
    size_to_idx = {size: i for i, size in enumerate(sorted(unique_sizes))}
    
    print("\n🚀 Running Ensemble Inference with 5x TTA (Original + Flips + Rotation)...")
    
    with torch.no_grad():
        # Iterate through all loaders simultaneously
        for batch_tuple in tqdm(zip(*loader_list), total=len(loader_list[0])):
            # batch_tuple[i] is (images, labels) for the i-th size
            
            # Check labels consistency across loaders
            labels_reference = batch_tuple[0][1]
            for i in range(1, len(batch_tuple)):
                if not torch.equal(labels_reference, batch_tuple[i][1]):
                    raise RuntimeError("Data ordering mismatch between loaders!")
            
            target_labels = labels_reference.to(DEVICE)
            
            # Ensemble Aggregation
            ensemble_probs = None
            total_weight = 0
            
            for model_info in loaded_models:
                # Select correct input for this model
                size = model_info['img_size']
                idx = size_to_idx[size]
                images = batch_tuple[idx][0].to(DEVICE)
                weight = model_info['weight']
                
                # --- TTA (Test Time Augmentation) ---
                # Initialize probabilities accumulator
                tta_probs_sum = None
                
                # 1. Original
                outputs_orig = model_info['model'](images)
                probs = torch.softmax(outputs_orig, dim=1)
                tta_probs_sum = probs
                
                # 2. Horizontal Flip
                images_hflip = torch.flip(images, [3])
                outputs = model_info['model'](images_hflip)
                probs = torch.softmax(outputs, dim=1)
                tta_probs_sum += probs
                
                # 3. Vertical Flip
                images_vflip = torch.flip(images, [2])
                outputs = model_info['model'](images_vflip)
                probs = torch.softmax(outputs, dim=1)
                tta_probs_sum += probs
                
                # 4. Rotate +15 degrees
                images_r15 = TF.rotate(images, 15)
                outputs = model_info['model'](images_r15)
                probs = torch.softmax(outputs, dim=1)
                tta_probs_sum += probs
                
                # 5. Rotate -15 degrees
                images_rn15 = TF.rotate(images, -15)
                outputs = model_info['model'](images_rn15)
                probs = torch.softmax(outputs, dim=1)
                tta_probs_sum += probs
                
                # Average TTA probs (5 augmentations)
                model_probs = tta_probs_sum / 5.0
                
                # Weighted accumulation for ensemble
                if ensemble_probs is None:
                    ensemble_probs = model_probs * weight
                else:
                    ensemble_probs += model_probs * weight
                total_weight += weight
            
            # Normalize probabilities
            ensemble_probs /= total_weight
            
            # Get Predictions
            _, preds = torch.max(ensemble_probs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target_labels.cpu().numpy())

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------
    accuracy = accuracy_score(all_labels, all_preds)
    
    print("\n" + "="*50)
    print(f"🏆 ENSEMBLE TEST ACCURACY: {accuracy*100:.2f}%")
    print("="*50)
    
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))
    
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    try:
        plot_confusion_matrix(cm, classes)
    except Exception as e:
        print(f"Could not save plot: {e}")

if __name__ == "__main__":
    main()
