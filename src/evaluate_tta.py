import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

from src.config import Config
from src.dataset import OralCancerDataset
from src.model import get_model
from src.utils import plot_confusion_matrix, print_classification_report


def get_tta_transforms():
    """Returns a list of different augmentation transforms for TTA"""
    base_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    tta_transforms = [
        # Original
        A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # Horizontal flip
        A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # Vertical flip
        A.Compose([
            A.Resize(224, 224),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # Both flips
        A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # Slight brightness
        A.Compose([
            A.Resize(224, 224),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # Rotation +15
        A.Compose([
            A.Resize(224, 224),
            A.Rotate(limit=15, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # Rotation -15
        A.Compose([
            A.Resize(224, 224),
            A.Rotate(limit=-15, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # CLAHE
        A.Compose([
            A.Resize(224, 224),
            A.CLAHE(clip_limit=4.0, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
    ]
    
    return tta_transforms


class TTADataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transforms_list):
        self.image_paths = image_paths
        self.labels = labels
        self.transforms_list = transforms_list
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image once
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Apply all TTA transforms
        augmented_images = []
        for transform in self.transforms_list:
            transformed = transform(image=image)
            augmented_images.append(transformed['image'])
        
        # Stack all augmented versions
        augmented_batch = torch.stack(augmented_images)
        
        return augmented_batch, label


def evaluate_with_tta(model, test_dir, device, tta_transforms):
    model.eval()
    
    # Load test images
    test_dataset = OralCancerDataset(test_dir, transform=None)
    image_paths = test_dataset.images
    labels = test_dataset.labels
    
    print(f"Test dataset size: {len(image_paths)}")
    print(f"Number of TTA transforms: {len(tta_transforms)}")
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(range(len(image_paths)), desc='TTA Evaluation')
        for idx in progress_bar:
            img_path = image_paths[idx]
            label = labels[idx]
            
            # Load image
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
            
            # Apply all TTA transforms and collect predictions
            tta_probs = []
            for transform in tta_transforms:
                transformed = transform(image=image)
                img_tensor = transformed['image'].unsqueeze(0).to(device)
                
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                tta_probs.append(probs.cpu().numpy())
            
            # Average all TTA predictions
            avg_probs = np.mean(tta_probs, axis=0)
            predicted = np.argmax(avg_probs, axis=1)[0]
            
            all_preds.append(predicted)
            all_labels.append(label)
            
            progress_bar.set_postfix({'acc': 100.*np.mean(np.array(all_preds) == np.array(all_labels))})
    
    return np.array(all_preds), np.array(all_labels)


def main():
    print("=" * 50)
    print("Model Evaluation with Test-Time Augmentation (TTA)")
    print("=" * 50)
    print(f"Device: {Config.DEVICE}")
    print(f"Model: {Config.BEST_MODEL_PATH}")
    print("=" * 50)
    
    # Load model
    model = get_model(
    num_classes=Config.NUM_CLASSES, 
    pretrained=False,
    model_name=Config.MODEL_NAME  # ✅ Add this
)
    checkpoint = torch.load(Config.BEST_MODEL_PATH, map_location=Config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully\n")
    
    # Get TTA transforms
    tta_transforms = get_tta_transforms()
    
    # Evaluate with TTA
    predictions, labels = evaluate_with_tta(model, Config.TEST_DIR, Config.DEVICE, tta_transforms)
    
    # Calculate metrics
    accuracy = 100. * (predictions == labels).sum() / len(labels)
    
    print("\n" + "=" * 50)
    print(f"TTA Test Accuracy: {accuracy:.2f}%")
    print("=" * 50)
    
    print_classification_report(labels, predictions, Config.CLASS_NAMES)
    plot_confusion_matrix(labels, predictions, Config.CLASS_NAMES, save_path='confusion_matrix_tta.png')


if __name__ == '__main__':
    main()
