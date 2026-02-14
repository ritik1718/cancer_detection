import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import timm
import torchvision.models as models
from torchvision.models import DenseNet121_Weights
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Add parent dir to path for src imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.dataset import OralCancerDataset, get_transforms

# ============================================
# HYBRID EVALUATION CONFIGURATION
# ============================================
IMG_SIZE = 384
BATCH_SIZE = 4
NUM_CLASSES = 2
CLASS_NAMES = ['Normal', 'OSCC']

# Data paths
DATA_DIR = r'c:\Projects\oral-cancer-detection\data\raw'
TEST_DIR = os.path.join(DATA_DIR, 'test')

# Model checkpoint path
MODEL_SAVE_DIR = r'c:\Projects\oral-cancer-detection\models\saved_models'
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'best_model_hybrid_vit_densenet.pth')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class HybridViTDenseNet(nn.Module):
    """
    Hybrid Model combining:
    - DenseNet121: Local feature extraction (textures, edges, local patterns)
    - ViT-Base-384: Global feature extraction (long-range dependencies, context)
    
    Architecture:
        Input (384x384) → [DenseNet Features] + [ViT CLS Token] → Fusion → Classifier
    """
    
    def __init__(self, num_classes=2, pretrained=False):
        super(HybridViTDenseNet, self).__init__()
        
        # 1. LOCAL FEATURES: DenseNet121
        if pretrained:
            self.densenet = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        else:
            self.densenet = models.densenet121(weights=None)
        
        self.densenet_features = self.densenet.features
        self.densenet_pooling = nn.AdaptiveAvgPool2d((1, 1))
        densenet_dim = 1024
        
        # 2. GLOBAL FEATURES: ViT-Base-384
        self.vit = timm.create_model('vit_base_patch16_384', pretrained=pretrained, num_classes=0)
        vit_dim = 768
        
        # 3. FUSION AND CLASSIFICATION
        self.fusion_dim = densenet_dim + vit_dim  # 1024 + 768 = 1792
        
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # --- DenseNet Branch (Local) ---
        dn_feat = self.densenet_features(x)
        dn_feat = self.densenet_pooling(dn_feat)
        dn_feat = torch.flatten(dn_feat, 1)
        
        # --- ViT Branch (Global) ---
        vit_feat = self.vit(x)
             
        # --- Fusion ---
        combined = torch.cat((dn_feat, vit_feat), dim=1)
        
        output = self.classifier(combined)
        return output


def evaluate_model(model, dataloader, device):
    """Run evaluation and collect predictions."""
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Evaluating')
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({'acc': f'{100.*correct/total:.2f}%'})
    
    accuracy = 100. * correct / total
    return np.array(all_preds), np.array(all_labels), accuracy


def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix_hybrid.png'):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix - Hybrid Model (ViT + DenseNet)')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f'📊 Confusion matrix saved to {save_path}')


def print_classification_report(y_true, y_pred, class_names):
    """Print detailed classification report."""
    print("\n" + "=" * 50)
    print("Classification Report")
    print("=" * 50)
    print(classification_report(y_true, y_pred, target_names=class_names))


def print_per_class_accuracy(y_true, y_pred, class_names):
    """Print per-class accuracy breakdown."""
    cm = confusion_matrix(y_true, y_pred)
    print("\n--- Per-Class Accuracy ---")
    for i, name in enumerate(class_names):
        if cm[i].sum() > 0:
            acc = 100.0 * cm[i][i] / cm[i].sum()
            print(f"  {name}: {acc:.2f}% ({cm[i][i]}/{cm[i].sum()})")


def main():
    print("=" * 60)
    print("🔬 HYBRID MODEL EVALUATION (ViT-Base-384 + DenseNet121)")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Model Path: {BEST_MODEL_PATH}")
    print(f"Test Dir: {TEST_DIR}")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists(BEST_MODEL_PATH):
        print(f"\n❌ Model checkpoint not found at: {BEST_MODEL_PATH}")
        print("   Please train the hybrid model first using hybrid_training.py")
        return
    
    # Load test dataset
    print("\n📁 Loading test dataset...")
    test_dataset = OralCancerDataset(
        TEST_DIR,
        transform=get_transforms(train=False, img_size=IMG_SIZE)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Initialize model (pretrained=False since we load weights from checkpoint)
    print("\n🧠 Loading Hybrid Model...")
    model = HybridViTDenseNet(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
    
    # Load trained weights
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Print checkpoint info
    if 'epoch' in checkpoint:
        print(f"  Checkpoint epoch: {checkpoint['epoch'] + 1}")
    if 'accuracy' in checkpoint:
        print(f"  Checkpoint val accuracy: {checkpoint['accuracy']:.2f}%")
    print("  ✅ Model loaded successfully\n")
    
    # Evaluate
    predictions, labels, accuracy = evaluate_model(model, test_loader, DEVICE)
    
    # Results
    print("\n" + "=" * 60)
    print(f"🎯 Test Accuracy: {accuracy:.2f}%")
    print("=" * 60)
    
    print_per_class_accuracy(labels, predictions, CLASS_NAMES)
    print_classification_report(labels, predictions, CLASS_NAMES)
    plot_confusion_matrix(labels, predictions, CLASS_NAMES)
    
    print("\n" + "=" * 60)
    print("✅ Evaluation Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
